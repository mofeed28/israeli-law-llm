from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data" / "raw"
KOLZCHUT_DIR = DATA_DIR / "kolzchut"
COURT_DIR = DATA_DIR / "court"
KNESSET_DIR = DATA_DIR / "knesset"

# Request settings
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds

# Kol-Zchut MediaWiki API
KOLZCHUT_API_URL = "https://www.kolzchut.org.il/w/api.php"

# Supreme Court decisions
COURT_BASE_URL = "https://supremedecisions.court.gov.il"
COURT_SEARCH_URL = f"{COURT_BASE_URL}/Home/SearchResult"

# Knesset OData API
KNESSET_API_URL = "https://knesset.gov.il/Odata/ParliamentInfo.svc"
KNESSET_LAWS_ENDPOINT = f"{KNESSET_API_URL}/KNS_Law"

# Hebrew Wikisource (full law texts - "Open Book of Laws")
WIKISOURCE_API_URL = "https://he.wikisource.org/w/api.php"
WIKISOURCE_DIR = DATA_DIR / "wikisource_laws"

# User agent
USER_AGENT = "IsraeliLawResearchBot/1.0 (Academic Research; LLM Fine-tuning)"
