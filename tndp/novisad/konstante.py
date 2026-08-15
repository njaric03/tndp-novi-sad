from pathlib import Path

# gde zavrsavaju sirovi i sredjeni podaci; ceo data/novisad/ je u.gitignore
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "novisad"
RAW = DATA / "raw"

GSP = "https://www.gspns.co.rs"
NSMART = "https://online.nsmart.rs"
OVERPASS = "https://overpass-api.de/api/interpreter"
NSINFO = "https://www.nsinfo.co.rs/lat"

# red voznje koji je bio na snazi kad su podaci preuzeti; sajt drzi samo tekucu verziju
VAZIOD = "2026-06-22"
DANI = {"R": "radni dan", "S": "subota", "N": "nedelja"}
REZIMI = {"rvg": "gradski", "rvp": "prigradski"}

# okvir oko Grada Novog Sada za Overpass upite (jug, zapad, sever, istok)
BBOX = (45.15, 19.60, 45.42, 20.05)

# uzi okvir za ulicnu mrezu i sadrzaje: izmeren raspon stajalista zone I (lat 45.175-45.337
BBOX_ULICE = (19.62, 45.15, 19.93, 45.36)

# zona I po GSP-ovom cenovniku (/cenovnik-sen/zona-mesta/1)
ZONA_I = [
    "ШАНГАЈ", "БУКОВАЦ", "РУМЕНКА", "ВЕТЕРНИК", "ПАРАГОВО", "ПОПОВИЦА",
    "СРЕМСКА КАМЕНИЦА", "ПЕТРОВАРАДИН", "ПЕЈИЋЕВИ САЛАШИ", "НОВИ ЛЕДИНЦИ",
    "НОВИ САД",
]

# broj voznji putnika po gradskim linijama, radni dan, sistemsko brojanje 2017
PUTNICI_2017 = {
    "1": 13120, "2": 14923, "3": 16851, "4": 15096, "5": 17412, "6": 10331,
    "7": 12711, "8": 18793, "9": 19879, "10": 497, "11": 9509, "12": 8809,
    "13": 5023, "14": 6504, "15": 2080, "16": 89, "17": 948, "18": 112,
}

# iz istog rada: vrsni sati i njihov udeo u dnevnom broju putovanja, broj vozila na radu, i ranija brojanja radi trenda
VRSNI_SAT_2017 = {"07-08": 14251, "13-14": 16130}
VOZILA_NA_RADU_2017 = 100
TREND_PUTNIKA = {2000: 223721, 2010: 181405, 2017: 172687}

# Popis stanovnistva 2022, RZS: 16 naselja Grada Novog Sada
POPIS_2022 = {
    "NOVI SAD": 260438, "VETERNIK": 18849, "FUTOG": 18011, "PETROVARADIN": 15621,
    "SREMSKA KAMENICA": 12632, "KAĆ": 11067, "RUMENKA": 6300, "KOVILJ": 5151,
    "KISAČ": 4511, "BUKOVAC": 3632, "BUDISAVA": 3107, "BEGEČ": 3005,
    "ČENEJ": 1942, "LEDINCI": 1864, "STEPANOVIĆEVO": 1848, "STARI LEDINCI": 985,
}

# OSM oznake za sadrzaje; kategorizacija preslikana iz njaric03/mu-novi-sad-tipologija-zgrada
POI_TAGOVI = ["shop", "amenity", "office", "leisure"]

# tezine kategorija u strani privlacnosti gravitacionog modela
TEZINE_SADRZAJA = {
    "office": 3.0, "education": 3.0, "health": 2.0, "retail": 1.5,
    "civic": 1.5, "food": 1.0, "leisure": 1.0, "grocery": 0.5,
}
