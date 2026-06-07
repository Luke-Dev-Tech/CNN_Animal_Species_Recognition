from flask import Flask, request, jsonify, render_template
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

CLASS_NAMES = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison',
               'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach',
               'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 
               'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
               'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 
               'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 
               'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 
               'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 
               'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes',
               'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 
               'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow',
               'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf',
               'wombat', 'woodpecker', 'zebra']

ANIMAL_EMOJIS = {
    "antelope": "🦌", "badger": "🦡", "bat": "🦇", "bear": "🐻",
    "bee": "🐝", "beetle": "🪲", "bison": "🦬", "boar": "🐗",
    "butterfly": "🦋", "cat": "🐱", "caterpillar": "🐛", "chimpanzee": "🐒",
    "cockroach": "🪳", "cow": "🐄", "coyote": "🐺", "crab": "🦀",
    "crow": "🐦", "deer": "🦌", "dog": "🐕", "dolphin": "🐬",
    "donkey": "🫏", "dragonfly": "🪲", "duck": "🦆", "eagle": "🦅",
    "elephant": "🐘", "flamingo": "🦩", "fly": "🪰", "fox": "🦊",
    "goat": "🐐", "goldfish": "🐠", "goose": "🪿", "gorilla": "🦍",
    "grasshopper": "🦗", "hamster": "🐹", "hare": "🐇", "hedgehog": "🦔",
    "hippopotamus": "🦛", "hornbill": "🐦", "horse": "🐴", "hummingbird": "🐦",
    "hyena": "🐺", "jellyfish": "🪼", "kangaroo": "🦘", "koala": "🐨",
    "ladybugs": "🐞", "leopard": "🐆", "lion": "🦁", "lizard": "🦎",
    "lobster": "🦞", "mosquito": "🦟", "moth": "🦋", "mouse": "🐭",
    "octopus": "🐙", "okapi": "🦒", "orangutan": "🦧", "otter": "🦦",
    "owl": "🦉", "ox": "🐂", "oyster": "🦪", "panda": "🐼",
    "parrot": "🦜", "pelecan": "🐦", "penguin": "🐧", "pig": "🐷",
    "pigeon": "🕊️", "porcupine": "🦔", "possum": "🐭", "raccoon": "🦝",
    "rat": "🐀", "reindeer": "🦌", "rhinoceros": "🦏", "sandpiper": "🐦",
    "seahorse": "🐴", "seal": "🦭", "shark": "🦈", "sheep": "🐑",
    "snake": "🐍", "sparrow": "🐦", "squid": "🦑", "squirrel": "🐿️",
    "starfish": "⭐", "swan": "🦢", "tiger": "🐯", "turkey": "🦃",
    "turtle": "🐢", "whale": "🐋", "wolf": "🐺", "wombat": "🐾",
    "woodpecker": "🐦", "zebra": "🦓"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    m = torchvision.models.efficientnet_b2(weights=None)
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=len(CLASS_NAMES), bias=True)
    )
    path = os.path.join(os.path.dirname(__file__), "Animals_Classifier.pth")
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m.to(device)

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1)[0]

        top_probs, top_indices = probs.topk(5)
        predictions = [
            {
                "label": CLASS_NAMES[idx.item()],
                "emoji": ANIMAL_EMOJIS.get(CLASS_NAMES[idx.item()], "🐾"),
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
