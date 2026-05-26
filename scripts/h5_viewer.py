import s3fs
import h5py

fs = s3fs.S3FileSystem(profile='stanford')

path = "bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5"

with fs.open(path, "rb") as f:
    with h5py.File(f, "r") as h5:
        seg = h5["segments"]
        print("segments type:", type(seg))
        print("segments keys:", list(seg.keys()))