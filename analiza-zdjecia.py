import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def wczytaj_zdjecie_z_url():
    while True:
        url = input("Podaj URL zdjęcia (.JPG): ").strip()
        try:
            with urllib.request.urlopen( url ) as plik:
                obraz = mpimg.imread(plik, format="jpg")
                print(f"Wczytano zdjęcie {obraz.shape[:2]}")
                return obraz
        except Exception as blad:
            print(f"Błąd: {blad}\nSpróbuj ponownie. \n")

def oszacuj_jakosc(obraz):
    skala_szarosci = cv2.cvtColor(obraz, cv2.COLOR_RGB2GRAY)
    histogram, _ = np.histogram(skala_szarosci, bins=256, range=(0, 256))
    n = skala_szarosci.size

    jasnosc = np.mean(skala_szarosci)
    kontrast = np.std(skala_szarosci)
    przyciete_cienie = np.sum(histogram[:4]) / n * 100
    przyciete_swiatlo = np.sum(histogram[252:]) / n * 100

    ocena = 100
    ocena -= abs( jasnosc - 128) * 0.2
    ocena -= np.maximum(0, 50 - kontrast) * 0.6
    ocena -= przyciete_cienie * 3
    ocena -= przyciete_swiatlo * 3
    ocena = np.clip(ocena, 0, 100)

    raport = (
        f"Jasność: {jasnosc:.1f}    (cel 128, -0,2 pkt/odchylenie)\n"
        f"Kontrast: {kontrast:.1f}    (cel >= 50, -0,6 pkt/odchylenie)\n"
        f"Przycięte cienie: {przyciete_cienie:.1f}%    (cel 0%, -3 pkt/%)\n"
        f"Przycięte światło: {przyciete_swiatlo:.1f}%    (cel 0%, -3 pkt/%)\n\n"
        f"Ocena: {ocena:.0f}/100\n"
    )

    return raport

def popraw_jakosc(obraz):
    obraz_temp = np.float32(obraz)
    
    skala_szarosci = cv2.cvtColor(obraz, cv2.COLOR_RGB2GRAY)
    kontrast = skala_szarosci.std()
    jasnosc = skala_szarosci.mean()

    if kontrast < 50:
        mnoznik = min(51 / kontrast, 1.4)
        obraz_temp = (obraz_temp - jasnosc) * mnoznik + jasnosc

    obraz_temp += np.clip(128 - jasnosc, -30, 30)
                      
    return np.clip(obraz_temp, 0, 255).astype(np.uint8)
    
obraz = wczytaj_zdjecie_z_url()
raport = oszacuj_jakosc(obraz)
skala_szarosci = cv2.cvtColor(obraz, cv2.COLOR_RGB2GRAY)

rysunek = plt.figure(figsize=(15, 8), layout="constrained")
osie = rysunek.subplot_mosaic(
    """
    OSR
    OGB
    """,
    width_ratios=[2, 1.3, 1.3]
)

osie['O'].imshow(obraz)
osie['O'].axis("off")
osie['O'].set_title(raport, loc="left")
osie['O'].set_aspect('equal', anchor='S')

osie['S'].hist(skala_szarosci.ravel(), bins=256, range=(0, 256), color='gray')
osie['S'].set_title('Skala szarości')

kanaly = [('R', 0, 'red', 'Kanał R'),
          ('G', 1, 'green', 'Kanał G'),
          ('B', 2, 'blue', 'Kanał B')]

for pozycja, nr_kanalu, kolor, tytul in kanaly:
    osie[pozycja].hist(obraz[:, :, nr_kanalu].ravel(), bins=256, range=(0, 256), color=kolor)
    osie[pozycja].set_title(tytul)

max_y = max(osie[pozycja].get_ylim()[1] for pozycja in ['S', 'R', 'G', 'B'])

for pozycja in ['S', 'R', 'G', 'B']:
    osie[pozycja].set_xlim(0, 255)
    osie[pozycja].set_ylim(0, max_y)

plt.show()

obraz_lepszy = popraw_jakosc(obraz)
raport_lepszego = oszacuj_jakosc(obraz_lepszy)

rysunek2, os2 = plt.subplots(1, 2, figsize=(16, 9))
os2[0].imshow(obraz)
os2[0].set_title(raport, loc="left")
os2[0].axis("off")

os2[1].imshow(obraz_lepszy)
os2[1].set_title(raport_lepszego, loc="left")
os2[1].axis("off")

plt.subplots_adjust(bottom=0, top=0.8)
plt.show()
