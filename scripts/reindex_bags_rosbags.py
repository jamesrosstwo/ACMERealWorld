"""Re-index RealSense .bag files by reading raw chunks and rewriting via rosbags.

Uses the pure-Python rosbags library's Writer (which handles INDEX_DATA, CHUNK_INFO
correctly). Parses the unindexed source bag manually (since rosbags.Reader refuses
to open unindexed bags), extracts connections and messages, then writes a clean bag.

Run with the ACMEReal env that has rosbags installed:

    /home/james/anaconda3/envs/ACMEReal/bin/python scripts/reindex_bags_rosbags.py <episode_dir>
"""
import struct
import sys
from pathlib import Path

import lz4.frame
from rosbags.rosbag1 import Writer

OP_MESSAGE_DATA = 0x02
OP_BAG_HEADER = 0x03
OP_INDEX_DATA = 0x04
OP_CHUNK = 0x05
OP_CHUNK_INFO = 0x06
OP_CONNECTION = 0x07

HEADER_MAGIC = b"#ROSBAG V2.0\n"


def read_header_fields(data: bytes) -> dict:
    fields = {}
    pos = 0
    while pos < len(data):
        field_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        field = data[pos : pos + field_len]
        pos += field_len
        sep = field.index(b"=")
        fields[field[: sep].decode()] = field[sep + 1 :]
    return fields


def parse_bag(bag_path: Path):
    """Yield (conn_info_dict, [(conn_id, timestamp_ns, data), ...]).

    conn_info_dict: conn_id -> {topic, msgtype, md5sum, msgdef, callerid, latching}
    """
    connections = {}
    messages = []

    with open(bag_path, "rb") as f:
        magic = f.read(len(HEADER_MAGIC))
        if magic != HEADER_MAGIC:
            raise ValueError(f"{bag_path} not a rosbag v2.0")

        hl = struct.unpack("<I", f.read(4))[0]
        f.read(hl)
        dl = struct.unpack("<I", f.read(4))[0]
        f.read(dl)

        while True:
            pos = f.tell()
            hl_bytes = f.read(4)
            if len(hl_bytes) < 4:
                break
            hl = struct.unpack("<I", hl_bytes)[0]
            hdr = f.read(hl)
            if len(hdr) < hl:
                break
            dl_bytes = f.read(4)
            if len(dl_bytes) < 4:
                break
            dl = struct.unpack("<I", dl_bytes)[0]
            fields = read_header_fields(hdr)
            op = struct.unpack("<B", fields.get("op", b"\x00"))[0]

            if op == OP_CHUNK:
                data = f.read(dl)
                if len(data) < dl:
                    break
                if fields.get("compression", b"none") == b"lz4":
                    try:
                        data = lz4.frame.decompress(data)
                    except Exception:
                        continue

                inner = 0
                while inner < len(data):
                    if inner + 4 > len(data):
                        break
                    ihl = struct.unpack_from("<I", data, inner)[0]
                    inner += 4
                    if inner + ihl > len(data):
                        break
                    ihdr_bytes = data[inner : inner + ihl]
                    inner += ihl
                    if inner + 4 > len(data):
                        break
                    idl = struct.unpack_from("<I", data, inner)[0]
                    inner += 4
                    idata = data[inner : inner + idl]
                    inner += idl

                    ihdr = read_header_fields(ihdr_bytes)
                    iop = struct.unpack("<B", ihdr["op"])[0]

                    if iop == OP_CONNECTION:
                        cid = struct.unpack("<I", ihdr["conn"])[0]
                        topic = ihdr["topic"].decode()
                        data_fields = read_header_fields(idata)
                        connections[cid] = {
                            "topic": topic,
                            "msgtype": data_fields["type"].decode(),
                            "md5sum": data_fields["md5sum"].decode(),
                            "msgdef": data_fields["message_definition"].decode(),
                            "callerid": data_fields.get("callerid", b"").decode() or None,
                            "latching": int(data_fields["latching"].decode()) if "latching" in data_fields else None,
                        }
                    elif iop == OP_MESSAGE_DATA:
                        cid = struct.unpack("<I", ihdr["conn"])[0]
                        t = struct.unpack("<Q", ihdr["time"])[0]
                        messages.append((cid, t, idata))
            else:
                f.seek(pos + 4 + hl + 4 + dl)

    return connections, messages


def reindex(bag_path: Path) -> bool:
    with open(bag_path, "rb") as f:
        if f.read(len(HEADER_MAGIC)) != HEADER_MAGIC:
            return False
        hl = struct.unpack("<I", f.read(4))[0]
        hdr = f.read(hl)
        bag_fields = read_header_fields(hdr)
        if struct.unpack("<Q", bag_fields["index_pos"])[0] != 0:
            print(f"  {bag_path.name}: already indexed, skipping")
            return False

    connections, messages = parse_bag(bag_path)
    if not messages:
        print(f"  {bag_path.name}: no messages parsed")
        return False

    tmp_path = bag_path.with_suffix(".bag.reindexed")
    if tmp_path.exists():
        tmp_path.unlink()

    w = Writer(tmp_path)
    w.set_compression(Writer.CompressionFormat.BZ2)
    with w:
        conn_map = {}
        for cid, info in connections.items():
            msgtype = info["msgtype"]
            if "/msg/" not in msgtype and msgtype.count("/") == 1:
                pkg, name = msgtype.split("/")
                msgtype = f"{pkg}/msg/{name}"
            conn_map[cid] = w.add_connection(
                topic=info["topic"],
                msgtype=msgtype,
                msgdef=info["msgdef"],
                md5sum=info["md5sum"],
                callerid=info["callerid"],
                latching=info["latching"],
            )
        for cid, t, data in messages:
            w.write(conn_map[cid], t, data)

    backup = bag_path.with_suffix(".bag.orig")
    bag_path.rename(backup)
    tmp_path.rename(bag_path)
    print(f"  {bag_path.name}: reindexed ({len(messages)} msgs, backup at {backup.name})")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <episode_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    bags = sorted(target.rglob("*.bag"))
    bags = [b for b in bags if not b.stem.endswith(".orig")]
    print(f"Found {len(bags)} bags")
    n_ok = 0
    for b in bags:
        try:
            if reindex(b):
                n_ok += 1
        except Exception as e:
            print(f"  {b.name}: FAILED: {e}")
    print(f"Reindexed {n_ok}/{len(bags)}")


if __name__ == "__main__":
    main()
