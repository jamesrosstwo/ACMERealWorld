"""Rewrite RealSense .bag files with proper index for pyrealsense2.

Copies valid chunks verbatim and rebuilds the index section.
Skips chunks that fail to decompress.

Usage::

    python scripts/rewrite_bags.py /path/to/episodes
"""
import struct
import sys
import lz4.frame
from pathlib import Path

HEADER_MAGIC = b"#ROSBAG V2.0\n"
OP_MESSAGE_DATA = 0x02
OP_BAG_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07


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
        try:
            sep = field.index(b"=")
            fields[field[:sep].decode("latin-1")] = field[sep + 1 :]
        except (ValueError, UnicodeDecodeError):
            break
    return fields


def make_header(fields):
    parts = []
    for k, v in fields.items():
        kv = k.encode() + b"=" + v
        parts.append(struct.pack("<I", len(kv)) + kv)
    return b"".join(parts)


def make_record(hdr_fields, data):
    hdr = make_header(hdr_fields)
    return struct.pack("<I", len(hdr)) + hdr + struct.pack("<I", len(data)) + data


def rewrite_bag(src_path: Path) -> bool:
    dst_path = src_path.with_suffix(".clean.bag")

    connections = {}
    chunks_info = []

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        magic = src.read(13)
        if magic != HEADER_MAGIC:
            print(f"  Not a rosbag: {src_path.name}")
            return False
        dst.write(magic)

        # Copy bag header
        hdr_len = read_uint32(src)
        hdr_data = src.read(hdr_len)
        data_len = read_uint32(src)
        padding_data = src.read(data_len)

        write_uint32(dst, hdr_len)
        dst.write(hdr_data)
        write_uint32(dst, data_len)
        dst.write(padding_data)

        chunk_count = 0
        skip_count = 0

        while True:
            raw = src.read(4)
            if len(raw) < 4:
                break
            rec_hdr_len = struct.unpack("<I", raw)[0]
            rec_hdr_data = src.read(rec_hdr_len)
            rec_data_len = read_uint32(src)

            fields = read_header_fields(rec_hdr_data)
            if "op" not in fields:
                src.seek(rec_data_len, 1)
                continue
            op = struct.unpack("<B", fields["op"])[0]

            if op == OP_CHUNK:
                chunk_raw = src.read(rec_data_len)
                comp = fields.get("compression", b"none").decode()

                try:
                    if comp == "lz4":
                        decompressed = lz4.frame.decompress(chunk_raw)
                    elif comp == "bz2":
                        import bz2
                        decompressed = bz2.decompress(chunk_raw)
                    else:
                        decompressed = chunk_raw
                except Exception:
                    skip_count += 1
                    continue

                # Write chunk to dst
                new_chunk_pos = dst.tell()
                write_uint32(dst, len(rec_hdr_data))
                dst.write(rec_hdr_data)
                write_uint32(dst, rec_data_len)
                dst.write(chunk_raw)

                # Parse inner records for index building
                msg_index = {}
                start_time = end_time = None
                ipos = 0
                while ipos < len(decompressed):
                    if ipos + 4 > len(decompressed):
                        break
                    ihl = struct.unpack_from("<I", decompressed, ipos)[0]
                    ipos += 4
                    if ipos + ihl > len(decompressed):
                        break
                    ihdr = read_header_fields(decompressed[ipos : ipos + ihl])
                    ipos += ihl
                    if ipos + 4 > len(decompressed):
                        break
                    idl = struct.unpack_from("<I", decompressed, ipos)[0]
                    ipos += 4
                    # Record the offset of this inner record within the decompressed chunk
                    inner_record_offset = ipos - idl - 4 - ihl - 4
                    ipos += idl

                    iop = struct.unpack("<B", ihdr.get("op", b"\x00"))[0]
                    if iop == OP_CONNECTION:
                        cid = struct.unpack("<I", ihdr["conn"])[0]
                        connections[cid] = (ihdr, decompressed[ipos - idl : ipos])
                    elif iop == OP_MESSAGE_DATA:
                        cid = struct.unpack("<I", ihdr["conn"])[0]
                        tv = struct.unpack("<Q", ihdr["time"])[0]
                        msg_index.setdefault(cid, []).append((tv, inner_record_offset))
                        if start_time is None or tv < start_time:
                            start_time = tv
                        if end_time is None or tv > end_time:
                            end_time = tv

                if msg_index:
                    chunks_info.append(
                        (new_chunk_pos, start_time or 0, end_time or 0, msg_index)
                    )
                chunk_count += 1

                # Write INDEX_DATA records after this chunk (standard rosbag1 layout)
                for cid, entries in sorted(msg_index.items()):
                    idx_hdr = {
                        "op": struct.pack("<B", OP_INDEX_DATA),
                        "ver": struct.pack("<I", 1),
                        "conn": struct.pack("<I", cid),
                        "count": struct.pack("<I", len(entries)),
                    }
                    idx_data = b""
                    for tv, offset in entries:
                        idx_data += struct.pack("<QI", tv, offset)
                    dst.write(make_record(idx_hdr, idx_data))
            else:
                # Skip non-chunk records (old index data, etc.)
                src.seek(rec_data_len, 1)

        # Write index section at end: CONNECTION + CHUNK_INFO
        index_start = dst.tell()

        for cid, (hdr, data) in sorted(connections.items()):
            dst.write(make_record(hdr, data))

        for chunk_pos, st, et, mi in chunks_info:
            cc = {cid: len(entries) for cid, entries in mi.items()}
            info_hdr = {
                "op": struct.pack("<B", OP_CHUNK_INFO),
                "ver": struct.pack("<I", 1),
                "chunk_pos": struct.pack("<Q", chunk_pos),
                "start_time": struct.pack("<Q", st),
                "end_time": struct.pack("<Q", et),
                "count": struct.pack("<I", len(cc)),
            }
            info_data = b"".join(
                struct.pack("<II", c, n) for c, n in sorted(cc.items())
            )
            dst.write(make_record(info_hdr, info_data))

        # Update bag header
        bag_hdr_fields = read_header_fields(hdr_data)
        bag_hdr_fields["index_pos"] = struct.pack("<Q", index_start)
        bag_hdr_fields["conn_count"] = struct.pack("<I", len(connections))
        bag_hdr_fields["chunk_count"] = struct.pack("<I", chunk_count)

        new_hdr = make_header(bag_hdr_fields)
        total_space = len(hdr_data) + len(padding_data)
        if len(new_hdr) > total_space:
            print(f"  ERROR: header too large for {src_path.name}")
            return False

        pad = total_space - len(new_hdr)
        dst.seek(13)
        write_uint32(dst, len(new_hdr))
        dst.write(new_hdr)
        write_uint32(dst, pad)
        dst.write(b"\x20" * pad)

    print(f"  {chunk_count} chunks, {skip_count} skipped, {len(connections)} conns")
    print(
        f"  {src_path.stat().st_size:,} -> {dst_path.stat().st_size:,} bytes"
    )

    # Replace original
    orig_backup = src_path.with_suffix(".orig.bag")
    if orig_backup.exists():
        orig_backup.unlink()
    src_path.rename(orig_backup)
    dst_path.rename(src_path)
    print(f"  Replaced {src_path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <episodes_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    bag_files = sorted(target.rglob("*.bag"))
    bag_files = [b for b in bag_files if not b.stem.endswith(".orig") and not b.stem.endswith(".clean")]
    if not bag_files:
        print(f"No .bag files found under {target}")
        sys.exit(1)

    print(f"Found {len(bag_files)} bag file(s)")
    ok = 0
    for bag in bag_files:
        print(f"Processing {bag.parent.name}/{bag.name}...")
        try:
            if rewrite_bag(bag):
                ok += 1
        except Exception as e:
            print(f"  Failed: {e}")
    print(f"\nRewrote {ok}/{len(bag_files)} bags")


if __name__ == "__main__":
    main()
