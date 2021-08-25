"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class vision_data_t(object):
    __slots__ = ["p_lidar", "lidar_quaternion"]

    def __init__(self):
        self.p_lidar = [ 0.0 for dim0 in range(3) ]
        self.lidar_quaternion = [ 0.0 for dim0 in range(4) ]

    def encode(self):
        buf = BytesIO()
        buf.write(vision_data_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack('>3d', *self.p_lidar[:3]))
        buf.write(struct.pack('>4d', *self.lidar_quaternion[:4]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != vision_data_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return vision_data_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = vision_data_t()
        self.p_lidar = struct.unpack('>3d', buf.read(24))
        self.lidar_quaternion = struct.unpack('>4d', buf.read(32))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if vision_data_t in parents: return 0
        tmphash = (0xe54d9d7346f5aed6) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if vision_data_t._packed_fingerprint is None:
            vision_data_t._packed_fingerprint = struct.pack(">Q", vision_data_t._get_hash_recursive([]))
        return vision_data_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

