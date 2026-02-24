"""Re-index RealSense .bag files by reading and rewriting them.

Usage::

    python scripts/reindex_bags.py /path/to/episodes
"""
import struct
import sys
from pathlib import Path


# rosbag v1 record op codes
OP_MESSAGE_DATA = 0x02
OP_BAG_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07

HEADER_MAGIC = b"#ROSBAG V2.0\n"


def read_uint32(f):
    return struct.unpack("<I", f.read(4))[0]


def write_uint32(f, v):
    f.write(struct.pack("<I", v))


def read_header_fields(data):
    fields = {}
    pos = 0
    while pos < len(data):
        field_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        field = data[pos : pos + field_len]
        pos += field_len
        sep = field.index(b"=")
        key = field[:sep].decode()
        value = field[sep + 1 :]
        fields[key] = value
    return fields


def make_header(fields: dict) -> bytes:
    parts = []
    for k, v in fields.items():
        kv = k.encode() + b"=" + v
        parts.append(struct.pack("<I", len(kv)) + kv)
    return b"".join(parts)


def make_record(header_fields: dict, data: bytes) -> bytes:
    hdr = make_header(header_fields)
    return struct.pack("<I", len(hdr)) + hdr + struct.pack("<I", len(data)) + data


def reindex_bag(bag_path: Path):
    with open(bag_path, "rb") as f:
        magic = f.read(len(HEADER_MAGIC))
        if magic != HEADER_MAGIC:
            print(f"  Skipping {bag_path.name}: not a rosbag v2.0 file")
            return False

        # Read BAG_HEADER
        bag_hdr_len = read_uint32(f)
        bag_hdr_data = f.read(bag_hdr_len)
        bag_hdr_fields = read_header_fields(bag_hdr_data)
        bag_data_len = read_uint32(f)
        f.read(bag_data_len)  # padding

        index_pos = struct.unpack("<Q", bag_hdr_fields["index_pos"])[0]
        if index_pos != 0:
            print(f"  Skipping {bag_path.name}: already indexed")
            return False

        # Scan chunks and collect index info
        connections = {}  # conn_id -> connection record fields + data
        chunks = []  # list of (chunk_file_pos, start_time, end_time, {conn_id: [(time, offset)]})

        while True:
            record_start = f.tell()
            hdr_len_bytes = f.read(4)
            if len(hdr_len_bytes) < 4:
                break
            hdr_len = struct.unpack("<I", hdr_len_bytes)[0]
            hdr_data = f.read(hdr_len)
            data_len = read_uint32(f)
            data_start = f.tell()

            fields = read_header_fields(hdr_data)
            if "op" not in fields:
                f.seek(data_start + data_len)
                continue
            op = struct.unpack("<B", fields["op"])[0]

            if op == OP_CHUNK:
                # Parse messages inside the chunk
                chunk_pos = record_start
                chunk_data = f.read(data_len)

                compression = fields.get("compression", b"none").decode()
                try:
                    if compression == "lz4":
                        import lz4.frame
                        chunk_data = lz4.frame.decompress(chunk_data)
                    elif compression == "bz2":
                        import bz2
                        chunk_data = bz2.decompress(chunk_data)
                except Exception:
                    # Skip chunks that fail to decompress
                    continue

                msg_index = {}  # conn_id -> [(time, offset)]
                start_time = None
                end_time = None
                inner_pos = 0

                while inner_pos < len(chunk_data):
                    if inner_pos + 4 > len(chunk_data):
                        break
                    inner_hdr_len = struct.unpack_from("<I", chunk_data, inner_pos)[0]
                    inner_pos += 4
                    if inner_pos + inner_hdr_len > len(chunk_data):
                        break
                    inner_hdr = read_header_fields(chunk_data[inner_pos : inner_pos + inner_hdr_len])
                    inner_pos += inner_hdr_len
                    if inner_pos + 4 > len(chunk_data):
                        break
                    inner_data_len = struct.unpack_from("<I", chunk_data, inner_pos)[0]
                    inner_pos += 4
                    inner_data = chunk_data[inner_pos : inner_pos + inner_data_len]
                    inner_pos += inner_data_len

                    inner_op = struct.unpack("<B", inner_hdr["op"])[0]

                    if inner_op == OP_CONNECTION:
                        conn_id = struct.unpack("<I", inner_hdr["conn"])[0]
                        connections[conn_id] = (inner_hdr, inner_data)

                    elif inner_op == OP_MESSAGE_DATA:
                        conn_id = struct.unpack("<I", inner_hdr["conn"])[0]
                        time_val = struct.unpack("<Q", inner_hdr["time"])[0]
                        if conn_id not in msg_index:
                            msg_index[conn_id] = []
                        # offset within the decompressed chunk data
                        msg_offset = inner_pos - inner_data_len - 4 - inner_hdr_len - 4
                        msg_index[conn_id].append((time_val, msg_offset))
                        if start_time is None or time_val < start_time:
                            start_time = time_val
                        if end_time is None or time_val > end_time:
                            end_time = time_val

                if msg_index:
                    chunks.append((chunk_pos, start_time or 0, end_time or 0, msg_index))
            else:
                f.seek(data_start + data_len)

    # Now rewrite: copy original up to end of chunks, then append index
    with open(bag_path, "rb") as f:
        f.seek(0, 2)
        file_end = f.tell()

    # Find end of last chunk (where we'll write the index)
    if not chunks:
        print(f"  Skipping {bag_path.name}: no chunks found")
        return False

    # Calculate where index should start: after the last chunk
    # Re-read to find end position of last chunk
    with open(bag_path, "rb") as f:
        last_chunk_pos = chunks[-1][0]
        f.seek(last_chunk_pos)
        hdr_len = read_uint32(f)
        f.seek(hdr_len, 1)
        data_len = read_uint32(f)
        f.seek(data_len, 1)
        index_start = f.tell()

    # Build index records
    index_records = []

    # Connection records (top-level)
    for conn_id, (hdr, data) in sorted(connections.items()):
        index_records.append(make_record(hdr, data))

    # Chunk info records (must come before index data for pyrealsense2)
    for chunk_pos, start_time, end_time, msg_index in chunks:
        conn_counts = {cid: len(entries) for cid, entries in msg_index.items()}
        info_hdr = {
            "op": struct.pack("<B", OP_CHUNK_INFO),
            "ver": struct.pack("<I", 1),
            "chunk_pos": struct.pack("<Q", chunk_pos),
            "start_time": struct.pack("<Q", start_time),
            "end_time": struct.pack("<Q", end_time),
            "count": struct.pack("<I", len(conn_counts)),
        }
        info_data = b""
        for cid, cnt in sorted(conn_counts.items()):
            info_data += struct.pack("<II", cid, cnt)
        index_records.append(make_record(info_hdr, info_data))

    # Truncate file at index_start and append index
    with open(bag_path, "r+b") as f:
        f.seek(index_start)
        f.truncate()
        for rec in index_records:
            f.write(rec)

        # Update BAG_HEADER index_pos
        f.seek(len(HEADER_MAGIC))
        bag_hdr_len = read_uint32(f)
        bag_hdr_data = f.read(bag_hdr_len)
        bag_hdr_fields = read_header_fields(bag_hdr_data)
        bag_hdr_fields["index_pos"] = struct.pack("<Q", index_start)
        bag_hdr_fields["conn_count"] = struct.pack("<I", len(connections))
        bag_hdr_fields["chunk_count"] = struct.pack("<I", len(chunks))
        new_hdr = make_header(bag_hdr_fields)

        # BAG_HEADER must stay the same size (padded)
        f.seek(len(HEADER_MAGIC))
        old_hdr_len = read_uint32(f)
        f.read(old_hdr_len)
        old_data_len = read_uint32(f)
        total_space = old_hdr_len + old_data_len

        if len(new_hdr) > total_space:
            print(f"  ERROR: new header too large for {bag_path.name}")
            return False

        padding = total_space - len(new_hdr)
        f.seek(len(HEADER_MAGIC))
        write_uint32(f, len(new_hdr))
        f.write(new_hdr)
        write_uint32(f, padding)
        f.write(b"\x20" * padding)

    print(f"  Reindexed {bag_path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <episodes_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    bag_files = sorted(target.rglob("*.bag"))
    if not bag_files:
        print(f"No .bag files found under {target}")
        sys.exit(1)

    print(f"Found {len(bag_files)} bag file(s)")
    reindexed = 0
    for bag in bag_files:
        try:
            if reindex_bag(bag):
                reindexed += 1
        except Exception as e:
            print(f"  Failed on {bag.name}: {e}")
    print(f"\nReindexed {reindexed}/{len(bag_files)} bags")


if __name__ == "__main__":
    main()
