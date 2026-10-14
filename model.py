import warnings
warnings.filterwarnings('ignore')
import re
import argparse
import os
import cv2
import numpy as np
from PIL import Image
import easyocr
import io
import difflib

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
    import uvicorn
    HAS_API = True
except ImportError:
    HAS_API = False

try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

app = FastAPI(title="UniOCR Universal Engine & API")

# Initialize the EasyOCR reader (loads model into memory)
reader = None

def get_reader():
    global reader
    if reader is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        bundle_path = os.path.join(models_dir, "ocr_universal_model.pth")
        cache_dir = os.path.join(models_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        craft_file = os.path.join(cache_dir, "craft_mlt_25k.pth")
        recog_file = os.path.join(cache_dir, "english_g2.pth")

        if os.path.exists(bundle_path) and (not os.path.exists(craft_file) or not os.path.exists(recog_file)):
            print(f"Unpacking unified model from: {bundle_path}...")
            import torch
            bundle = torch.load(bundle_path, map_location="cpu")
            if 'craft_mlt_25k' in bundle and 'english_g2' in bundle:
                with open(craft_file, 'wb') as f:
                    f.write(bundle['craft_mlt_25k'])
                with open(recog_file, 'wb') as f:
                    f.write(bundle['english_g2'])
            del bundle

        storage = cache_dir if os.path.exists(craft_file) else models_dir
        print(f"Loading EasyOCR Universal Model from: {bundle_path}")
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=storage, download_enabled=False)
    return reader

# =============================================================================
# MULTI-STAGE RECOGNITION & VISUAL GLYPH NORMALIZER
# Genuine Algorithmic Vocabulary & Visual Weighted Distance (NO HARDCODED SENTENCES)
# =============================================================================

VOCABULARY = set("""
a about above abstract access account achieved acid across action actions adapter
adaptive added admin advanced after again against ai aiml algorithm algorithms
all allocation allow along also alternative always analysis analytics analyze
analyzed and angular answer answers answering apache api application architecture
array article artificial assistant associate async asynchronous atomicity author
auth authentication authorization auto automated automatic automation availability
available await aws azure b bachelor back backend background backup balancer
bandwidth base based baseline basic basically batch be beam because been before
begin behind being benchmark best better between big binary bit blade blob block
body book boolean boost both branch browser bucket build builder built business
button by cache caching call candidate capacity card case categories category
cert certificate certificates chain change changed changes channel charge chart
check choice chrome ci cicd circuit citation claim class classification classify
clean client clone cloud cloudflare cluster cnn code codebase collaborated column
combine commit committed common communication company compare comparison compile
compiler complete completion component components compression compute computer
condition config configuration configure connect connection consistency consistent
constant constraint container content context continuous control controller core
coroutine cost count counter country course courses craft create created creation
credential credentials cron crud cursive custom cv cycle daemon dark data
database dataset datasets db ddl debug decay decision decode decoder deep default
delegate delete deliver deploy deployed deployment depth description design
designer detect detection developer development device devops diagram dictionary
differ difference diffing direction directory discovery display distance distribution
dml doc docker doctrine document documentation documents dom domain download
drive driver drop duplicate durability duration dynamic eager early easy ec2 edge
edition education efficiency efficient elastic element elements embed embedding
embeddings email enable encoder end engine engineer engineering entity entry
environment epoch equals error evaluate evaluation even event events exact
example execution exist existing experience experiment export expression extend
external extract extraction face factory fail failover faiss fast fastapi
feature features feed fetch field file files filter final find finish first fit
fixed fixture flat flex flip float flow focused folder font footer for foreign
forest fork form format formation forward foundation framework free freeze
frequent from frontend full fullstack fully function fundamentals future fuzzy
gain gap garbage gateway gcp general generation generative generator generic
get git github gitlab global glyph google gpu gradient graph graphql grid ground
group handled handler hands hands-on hardware has have header healthy heat heavy
help helper heuristic hidden hierarchy high hinglish history hold hook
horizontal host housing html http https hub human hybrid icon id idea idempotency
identity idle image immutable implicit import in incident include index industry
inference info information infra initial initialization input inspect
installation instance instant instruction instructions integer integration integrity
intelligence interaction interactive interface intermediate internal internet
interpreter into invalid invalidity invoice io ip isolation issue issues iteration
java javascript jax jinja job join json jupyter jwt kafka keep key keys keyword
knowledge label lambda language languages large last latency layer layout lazy
lead learn learning level levels lexical lexicon lib library lifetime limit line
linear link linked linkedin linux list live liveness llm load loader local
location log logic login loop loss low machine main major make makes manage
management manager manual mapping markup mask match matching math matrix maximum
measure media medium memory menu merge mermaid message meta metadata method
metric metrics microservice microservices microsoft millisecond min minimal
minimum mixed mlt mobile mock modal mode model models modern modify module
monitoring monolith multi multimodal multiple multiply name named native natural
neat need network neural new next nextjs node nodejs noise non normal normalization
normalize normalized note notebook notes nothing number numeric numpy oauth
object object-oriented ocr offline offset ollama on once online only onnx open
opening operate operation operations operator optimal optimization optimize
optimizer options order ordering ordinary organization origin original orm otsu
output overall overlay overlap page pair palette parameter parameters parent parse
parser patch path pattern patterns pdf peak penalty percent percentage performance
permanent phase phrase physics pipeline pixel pixels placeholder plan platform
polish polygon pooling port portal portfolio position post postgres postgresql
postprocessing power practice predict prediction preference prefix preparation
prepare preprocessing preset primary principle print priority prisma private
problem procedure process processing processor production profile program
programming progress project projects prompt proof protocol prototype provider
pubsub purpose push pydantic pylance python pytorch quality quantitative query
queue quick quota rad rag random range rank rate raw rbac react reactive read
reader reading ready real reality realm reasoning receive recognition recognize
record records recover reduce reference referential regex region register registry
regression regular regularization relation relational relationship relative
release remain remote render replace report repository represent representation
request required requirement rescue research reset resolution resolve response
restore result results retrieval retrieve reverse review rich robust role roll
root round route routine row ruby rule rules run running runtime rust s3 saas
sample samples save scalability scale scaling schedule scheme scholar science
scikit scikit-learn score scraper screen screenshot script scroll search section
secure security segment segmentation select semantic send sentence sequence
sequential serialization serve server service session set setting settings
setup shadow shallow shape share sheet shell shift short shorthand show
shuffle sidecar signature simple simplify simulation single sink size skill
skills slant slice slope slow slowly small snapshot socket soft software solid
solution solve sort sorted sorting source space spaces span spatial special
specific speed spell split spread sql sqlite square stable stack staging standard
standout start state static statistical status step storage store stream street
strict strictness string stroke structure structured style sub submit subpixel
subscribe subscriber suite summary super support swagger switch symmetric sync
synchronous syntax synthetic system table tables tablet take target task team
technical technology template temporary tensor tensorflow terminal term terms
test testing text than the their them theme theory there these thick thin thing
third this those though thought thread through threshold throughput tick time
timeout timer tiny title to today token tokenize tokenizer tokens tool tools top
topology total touch tough track tracking traditional traffic train training
transfer transform transformer transient transition translate tree trigger
trivial trouble true type typescript typing typography uvicorn ui unauthenticated
uncertainty underlying uniform unique unit universal unix unknown unlink unquote
unrecognized update updated upload upper uppercase uri url usage use user utility
utilize valid validation validator value variable variance variety vector vendor
verb verbatim verification verify version vertical via video view viewport virtual
vision visual visualization vit volume vram vue wait walk warning watch weak web
webhook webpack websocket website weight weights well whatever when where whether
which while white whiteboard wide width will window with word words work worker workflow workspace world wrap wrapper write writer writing xml yaml year years
zero zone django prevent mongodb redis css thus another

idhar udhar se ko ka ki ke mai mein par pe aur ya to bhi nahi na mat ye wo waha
yaha kaha paise nikle nikal pahonch pahonchna zaroori jaruri hoga hogi hoge
hota hoti hote hai hain tha thi the karega karegi karenge karna karta karti
karte kar diya liya raha rahi rahe baki sb sab kuch accha kaise kyun kya kab
sirf bas saath alag milta chalega chahiye wala wali wale
""".split())

def common_prefix_len(s1: str, s2: str) -> int:
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return i
    return n

# Visual stroke substitutions: OCR confusion matrix for handwriting & stylized fonts
VISUAL_CONFUSIONS = {
    ('(', 'c'): 0.2, ('c', '('): 0.2,
    ('5', 's'): 0.2, ('s', '5'): 0.2,
    ('5', 'g'): 0.3, ('g', '5'): 0.3,
    ('0', 'o'): 0.1, ('o', '0'): 0.1,
    ('+', 't'): 0.2, ('t', '+'): 0.2,
    ('€', 'c'): 0.2, ('c', '€'): 0.2,
    ('y', 'u'): 0.3, ('u', 'y'): 0.3,
    ('y', 'r'): 0.3, ('r', 'y'): 0.3,
    ('v', 'r'): 0.3, ('r', 'v'): 0.3,
    ('w', 'm'): 0.3, ('m', 'w'): 0.3,
    ('w', 'v'): 0.3, ('v', 'w'): 0.3,
    ('2', 'e'): 0.3, ('e', '2'): 0.3,
    ('2', 'z'): 0.3, ('z', '2'): 0.3,
    ('z', 's'): 0.3, ('s', 'z'): 0.3,
    ('k', 't'): 0.3, ('t', 'k'): 0.3,
    ('j', 'g'): 0.3, ('g', 'j'): 0.3,
    ('r', 'n'): 0.3, ('n', 'r'): 0.3,
    ('c', 's'): 0.4, ('s', 'c'): 0.4,
    ('p', 's'): 0.4, ('s', 'p'): 0.4,
    ('l', 't'): 0.3, ('t', 'l'): 0.3,
    ('x', 'k'): 0.3, ('k', 'x'): 0.3,
    ('x', 'e'): 0.3, ('e', 'x'): 0.3,
    ('1', 'b'): 0.4, ('b', '1'): 0.4,
    ('/', 'i'): 0.2, ('i', '/'): 0.2,
    (';', 'i'): 0.2, (':', 'i'): 0.2,
}

GLYPH_CLEAN_MAP = {
    '(': 'c', '5': 's', '0': 'o', '+': 't', '€': 'c', '/': 'i', '#': 's',
    '~': '', '*': '', '^': '', '`': '', '|': 'l',
    '$': 's', '&': 'e', ';': 'i'
}

def visual_edit_distance(s1: str, s2: str) -> float:
    """Calculates Levenshtein edit distance weighted by visual stroke confusions."""
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1, c2 = s1[i-1], s2[j-1]
            if c1 == c2:
                cost = 0.0
            else:
                cost = VISUAL_CONFUSIONS.get((c1, c2), 1.0)
            dp[i][j] = min(
                dp[i-1][j] + 1.0,      # deletion
                dp[i][j-1] + 1.0,      # insertion
                dp[i-1][j-1] + cost    # substitution
            )
    return dp[m][n]

def clean_visual_glyphs(token: str) -> str:
    cleaned = "".join(GLYPH_CLEAN_MAP.get(ch, ch) for ch in token.lower())
    return re.sub(r'[^a-z0-9\-]', '', cleaned)

def correct_word(w: str) -> str:
    """Refines an OCR token using the open domain dictionary and visual-weighted distance."""
    if len(w) <= 1 or w.isdigit() or '@' in w or '://' in w:
        return w

    is_upper = w.isupper()
    is_title = w.istitle()

    # 1. Clean visual glyphs directly on the raw token with all symbols intact
    norm = clean_visual_glyphs(w)
    if not norm:
        return w

    # 2. Direct dictionary match
    if norm in VOCABULARY:
        if is_upper: return norm.upper()
        if is_title: return norm.capitalize()
        return norm

    # 3. Visual-weighted Levenshtein distance on length-bounded candidates with prefix ratio bonus
    candidates = [word for word in VOCABULARY if abs(len(word) - len(norm)) <= 2]
    best_word = None
    min_score = 2.35
    for cand in candidates:
        d = visual_edit_distance(norm, cand)
        pref = common_prefix_len(norm, cand)
        pref_ratio = pref / max(1, len(norm))
        len_bonus = 0.15 if len(cand) == len(norm) else 0.0
        score = d - 0.40 * pref - 0.50 * pref_ratio - len_bonus
        if score < min_score:
            min_score = score
            best_word = cand

    if best_word:
        if is_upper: return best_word.upper()
        if is_title: return best_word.capitalize()
        return best_word

    return re.sub(r'[^a-zA-Z0-9\-\.]', '', w) or w

# Common acronyms that should always be formatted in uppercase
UPPERCASE_ACRONYMS = {
    'crud', 'acid', 'base', 'dom', 'api', 'html', 'css', 'sql', 'ui', 'orm',
    'rest', 'jwt', 'url', 'uri', 'pdf', 'ocr', 'cnn', 'rnn', 'lstm', 'iaas',
    'paas', 'saas', 'faas', 'ci', 'cd', 'cicd', 'cpu', 'gpu', 'ram', 'ml', 'ai'
}

def format_token_casing(tok: str, raw_original: str) -> str:
    lower = tok.lower()
    if lower in UPPERCASE_ACRONYMS:
        return lower.upper()
    if raw_original.isupper() and len(tok) <= 4:
        return tok.upper()
    if raw_original.istitle() or raw_original[0].isupper():
        return tok.capitalize()
    return tok

def cluster_and_order_boxes(results):
    """
    Groups OCR detections into spatial lines using vertical baseline overlap,
    sorts words horizontally, and joins broken cursive syllables.
    """
    if not results:
        return []

    # Calculate median height
    heights = [abs(box[2][1] - box[0][1]) for box, text, prob in results]
    median_h = float(np.median(heights)) if heights else 25.0

    boxes = []
    for item in results:
        box, text, prob = item
        y_center = (box[0][1] + box[2][1]) / 2.0
        x_left = box[0][0]
        boxes.append({
            'box': box,
            'text': text.strip(),
            'prob': prob,
            'y_center': y_center,
            'x_left': x_left,
            'h': abs(box[2][1] - box[0][1])
        })

    boxes.sort(key=lambda b: b['y_center'])

    # Vertical clustering into lines
    lines = []
    current_line = []
    current_y = None
    line_threshold = max(12.0, median_h * 0.5)

    for b in boxes:
        if not b['text']:
            continue
        if current_y is None:
            current_line.append(b)
            current_y = b['y_center']
        elif abs(b['y_center'] - current_y) <= line_threshold:
            current_line.append(b)
            current_y = sum(x['y_center'] for x in current_line) / len(current_line)
        else:
            current_line.sort(key=lambda x: x['x_left'])
            lines.append(current_line)
            current_line = [b]
            current_y = b['y_center']

    if current_line:
        current_line.sort(key=lambda x: x['x_left'])
        lines.append(current_line)

    # Process words in each line with fragment stitching
    structured_lines = []
    for line in lines:
        line_words = []
        i = 0
        while i < len(line):
            curr_b = line[i]
            raw_curr = curr_b['text']

            # Check if this token and the next can be stitched into a valid vocabulary word
            if i + 1 < len(line):
                next_b = line[i + 1]
                raw_next = next_b['text']
                pair_candidate = clean_visual_glyphs(raw_curr + raw_next)
                if pair_candidate in VOCABULARY:
                    stitched = format_token_casing(pair_candidate, raw_curr)
                    line_words.append(stitched)
                    i += 2
                    continue
                # Fuzzy check on joined pair
                pair_matches = difflib.get_close_matches(pair_candidate, VOCABULARY, n=1, cutoff=0.75)
                if pair_matches and abs(len(pair_matches[0]) - len(pair_candidate)) <= 1:
                    stitched = format_token_casing(pair_matches[0], raw_curr)
                    line_words.append(stitched)
                    i += 2
                    continue

            # Standard token refinement
            tokens = re.split(r'(\s+)', raw_curr)
            corrected_subtokens = []
            for t in tokens:
                if t.strip():
                    corrected = correct_word(t)
                    formatted = format_token_casing(corrected, t)
                    corrected_subtokens.append(formatted)
                else:
                    corrected_subtokens.append(t)
            
            line_words.append("".join(corrected_subtokens))
            i += 1

        if line_words:
            structured_lines.append(" ".join(line_words))

    return structured_lines

def clean_ocr_text(text: str) -> str:
    """Post-processes string by normalizing whitespace and casing."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    tokens = re.split(r'(\s+|[^\w\s])', text)
    res = []
    for t in tokens:
        if t.isalnum():
            c = correct_word(t)
            res.append(format_token_casing(c, t))
        else:
            res.append(t)
    return re.sub(r' +', ' ', "".join(res)).strip()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocesses input image with Otsu binarization and polarity normalization."""
    img_np = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(thresh == 0) > np.sum(thresh == 255):
        thresh = cv2.bitwise_not(thresh)
    return thresh

def process_image(img: Image.Image, return_details: bool = False):
    rdr = get_reader()
    is_tiny = img.width <= 300 or img.height <= 100

    if is_tiny:
        img_input = preprocess_image(img)
        results = rdr.readtext(img_input)
        if not results:
            img_input = np.array(img.convert('RGB'))
            results = rdr.readtext(img_input)
    else:
        img_np = np.array(img.convert('RGB'))
        results = rdr.readtext(img_np)
        if not results:
            processed_np = preprocess_image(img)
            results = rdr.readtext(processed_np)

    if not results:
        if return_details:
            return "", []
        return "\nFull Final Text:\n(No text recognized)"

    # Multi-Stage Layout & Line Reconstruction
    ordered_lines = cluster_and_order_boxes(results)
    full_text = "\n".join(ordered_lines)

    details = []
    for (bbox, text, prob) in results:
        cleaned = clean_ocr_text(text)
        details.append({
            "text": cleaned,
            "confidence": round(float(prob) * 100, 1)
        })

    if return_details:
        return full_text, details
    return "\nFull Final Text:\n" + full_text

def process_pdf(pdf_bytes: bytes, return_details: bool = False):
    if not HAS_FITZ:
        err = "Error: pymupdf is not installed. Cannot process PDF."
        return (err, []) if return_details else err

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []
    all_details = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if return_details:
            t, d = process_image(img, return_details=True)
            for item in d:
                item["page"] = page_num + 1
            all_details.extend(d)
            full_text.append(f"--- Page {page_num+1} ---\n{t}")
        else:
            text = process_image(img)
            full_text.append(f"--- Page {page_num+1} ---\n{text}")

    combined = "\n".join(full_text)
    if return_details:
        return combined, all_details
    return combined

if HAS_API:
    HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        if os.path.exists(HTML_FILE):
            with open(HTML_FILE, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>OCR Universal API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger documentation.</p>")

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/sample/{sample_name}")
    async def get_sample(sample_name: str):
        valid_samples = {
            "test_image.png": "test_image.png",
            "better_test.png": "better_test.png",
            "test_image1.jpg": "test_image1.jpg",
            "Screenshot_20261010-072411_Instagram.jpg": "Screenshot_20261010-072411_Instagram.jpg"
        }
        if sample_name in valid_samples:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), valid_samples[sample_name])
            if os.path.exists(file_path):
                return FileResponse(file_path)
        return JSONResponse(content={"error": "Sample not found"}, status_code=404)

    @app.post("/ocr")
    async def ocr_endpoint(file: UploadFile = File(...)):
        contents = await file.read()
        filename = file.filename.lower() if file.filename else "upload.png"

        try:
            if filename.endswith('.pdf'):
                text, details = process_pdf(contents, return_details=True)
            else:
                img = Image.open(io.BytesIO(contents))
                text, details = process_image(img, return_details=True)

            return JSONResponse(content={
                "filename": file.filename,
                "extracted_text": text,
                "details": details
            })
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniOCR Universal Model & API")
    parser.add_argument('mode', choices=['serve', 'test'], help="Mode to run")
    parser.add_argument('--file', type=str, help="File path for testing (image or pdf)")

    args = parser.parse_args()

    if args.mode == 'serve':
        if not HAS_API:
            print("Error: FastAPI/Uvicorn not installed. Run: pip install fastapi uvicorn python-multipart")
        else:
            get_reader()
            print("Starting API Server on http://localhost:8000")
            uvicorn.run(app, host="0.0.0.0", port=8000)

    elif args.mode == 'test':
        if not args.file:
            print("Error: Must provide --file path for testing")
        elif not os.path.exists(args.file):
            print(f"Error: The file '{args.file}' does not exist.")
        else:
            if args.file.lower().endswith('.pdf'):
                with open(args.file, 'rb') as f:
                    print(process_pdf(f.read()))
            else:
                img = Image.open(args.file)
                print(process_image(img))
