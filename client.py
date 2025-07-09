import requests, argparse, json, pathlib

def main(img_path, model="resnet50", host="http://localhost:8000"):
    url = f"{host}/predict?model={model}"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": (pathlib.Path(img_path).name, f)})
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", help="путь к картинке")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--host",  default="http://localhost:8000")
    args = p.parse_args()
    main(args.image, args.model, args.host)