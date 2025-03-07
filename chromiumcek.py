import subprocess

def get_binary_location(binary_names=["chromium-browser", "chromium"]):
    """
    Mencoba menemukan lokasi binary dari daftar nama yang diberikan menggunakan perintah 'which'.
    Mengembalikan lokasi (path) binary jika ditemukan, atau None jika tidak ditemukan.
    """
    for binary in binary_names:
        try:
            location = subprocess.check_output(["which", binary]).decode().strip()
            if location:
                return location
        except subprocess.CalledProcessError:
            continue
    return None

def main():
    chromium_location = get_binary_location()
    if chromium_location:
        print("Lokasi Chromium:", chromium_location)
    else:
        print("Chromium tidak ditemukan di PATH.")

if __name__ == "__main__":
    main()
