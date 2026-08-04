from pathlib import Path

# gde završavaju sirovi i sređeni podaci; ceo data/novisad/ je u .gitignore.
# putanja se računa od fajla, ne od radnog direktorijuma — inače pokretanje
# iz bilo čega osim korena repoa tiho napravi prazan data/novisad/ pored sebe
# umesto da nađe postojeći
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "novisad"
RAW = DATA / "raw"

GSP = "https://www.gspns.co.rs"
NSMART = "https://online.nsmart.rs"
OVERPASS = "https://overpass-api.de/api/interpreter"
NSINFO = "https://www.nsinfo.co.rs/lat"

# red vožnje koji je bio na snazi kad su podaci preuzeti; sajt drži samo tekuću
# verziju, stari datumi vraćaju praznu listu linija
VAZIOD = "2026-06-22"
DANI = {"R": "radni dan", "S": "subota", "N": "nedelja"}
REZIMI = {"rvg": "gradski", "rvp": "prigradski"}

# okvir oko Grada Novog Sada za Overpass upite (jug, zapad, sever, istok)
BBOX = (45.15, 19.60, 45.42, 20.05)

# uži okvir za uličnu mrežu i sadržaje: izmeren raspon stajališta zone I
# (lat 45.175-45.337, lon 19.648-19.901) uvećan za ~2.5 km sa svake strane.
# osmnx traži (zapad, jug, istok, sever).
BBOX_ULICE = (19.62, 45.15, 19.93, 45.36)

# zona I po GSP-ovom cenovniku (/cenovnik-sen/zona-mesta/1), u ćirilici kako je
# na sajtu; ovo je zvanična granica područja studije
ZONA_I = [
    "ШАНГАЈ", "БУКОВАЦ", "РУМЕНКА", "ВЕТЕРНИК", "ПАРАГОВО", "ПОПОВИЦА",
    "СРЕМСКА КАМЕНИЦА", "ПЕТРОВАРАДИН", "ПЕЈИЋЕВИ САЛАШИ", "НОВИ ЛЕДИНЦИ",
    "НОВИ САД",
]

# broj vožnji putnika po gradskim linijama, radni dan, sistemsko brojanje 2017.
# izvor: Lazarević, Pitka (2020), Zbornik radova FTN, doi 10.24867/06DS03Lazarevic,
# slika 1; podaci potiču iz studije FTN "Smart plan - prikupljanje podataka
# 'prva faza' - Istraživanje u javnom gradskom prevozu putnika", Novi Sad 2017.
# suma je 172.687 i poklapa se sa brojem koji rad navodi u tekstu.
PUTNICI_2017 = {
    "1": 13120, "2": 14923, "3": 16851, "4": 15096, "5": 17412, "6": 10331,
    "7": 12711, "8": 18793, "9": 19879, "10": 497, "11": 9509, "12": 8809,
    "13": 5023, "14": 6504, "15": 2080, "16": 89, "17": 948, "18": 112,
}

# iz istog rada: vršni sati i njihov udeo u dnevnom broju putovanja, broj vozila
# na radu, i ranija brojanja radi trenda
VRSNI_SAT_2017 = {"07-08": 14251, "13-14": 16130}
VOZILA_NA_RADU_2017 = 100
TREND_PUTNIKA = {2000: 223721, 2010: 181405, 2017: 172687}

# Popis stanovništva 2022, RZS: 16 naselja Grada Novog Sada. služi da se nivoi iz
# evidencije JKP Informatika (koja broji prijavljene, ne stanovnike) spuste na
# popisne. suma ovih brojeva je 368.963, a RZS za Grad objavljuje 368.967 —
# razlika od 4 stanovnika je bez uticaja na mase zona.
POPIS_2022 = {
    "NOVI SAD": 260438, "VETERNIK": 18849, "FUTOG": 18011, "PETROVARADIN": 15621,
    "SREMSKA KAMENICA": 12632, "KAĆ": 11067, "RUMENKA": 6300, "KOVILJ": 5151,
    "KISAČ": 4511, "BUKOVAC": 3632, "BUDISAVA": 3107, "BEGEČ": 3005,
    "ČENEJ": 1942, "LEDINCI": 1864, "STEPANOVIĆEVO": 1848, "STARI LEDINCI": 985,
}

# OSM oznake za sadržaje; kategorizacija preslikana iz njaric03/mu-novi-sad-tipologija-zgrada
POI_TAGOVI = ["shop", "amenity", "office", "leisure"]

# težine kategorija u strani privlačnosti gravitacionog modela. u tipologiji
# zgrada su kategorije bile ravnopravne jer se opisivala izgrađenost; ovde
# opisuju koliko sadržaj privlači putovanje autobusom, pa radna mesta i škole
# nose najviše a prodavnice hrane najmanje (uglavnom pešačka putovanja).
# ovo su pretpostavke i idu u analizu osetljivosti.
TEZINE_SADRZAJA = {
    "office": 3.0, "education": 3.0, "health": 2.0, "retail": 1.5,
    "civic": 1.5, "food": 1.0, "leisure": 1.0, "grocery": 0.5,
}
