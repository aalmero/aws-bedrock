import os
from urllib.request import urlretrieve

def main():
    os.makedirs("data", exist_ok=True)

    files = [
        "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
        "https://www.irs.gov/pub/irs-pdf/p15.pdf",
        "https://www.irs.gov/pub/irs-pdf/p1212.pdf"
    ]

    for url in files:
        file_path = os.path.join("data", url.rpartition("/")[2])
        urlretrieve(url, file_path)

if __name__ == "__main__":
    main()    