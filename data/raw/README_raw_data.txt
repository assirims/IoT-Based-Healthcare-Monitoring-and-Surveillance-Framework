Large archives are stored as 15 MB parts (name.zip.part01, part02, ...).
01_download_data.py joins them automatically (or: copy /b a.part01+a.part02 a.zip on Windows, cat a.part* > a.zip on Linux/macOS).
SHA256SUMS.txt lists the checksums of the original, re-assembled archives.
falls_selected/ holds the accelerometer and gyroscope channels of the UCI Simulated Falls and ADL dataset (one file per volunteer),
extracted from the original 1.2 GB archive by 04_prepare_falls.py (no names or contact data are included).
