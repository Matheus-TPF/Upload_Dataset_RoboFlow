import os
import json
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from roboflow import Roboflow
from tqdm import tqdm

load_dotenv()

CONFIG = {
    "API_KEY":   os.getenv("API_KEY"),
    "WORKSPACE": os.getenv("WORKSPACE"),
    "PROJECT":   os.getenv("PROJECT"),
    "PATHS": {
        "train": Path(os.getenv("CAMINHO_TREINAMENTO", "")),
        "valid": Path(os.getenv("CAMINHO_VALIDACAO",   "")),
        "test":  Path(os.getenv("CAMINHO_TESTE",       "")),
    },
    "IMG_EXTS":       {".jpg", ".jpeg", ".png"},
    "BATCH_SIZE":     50,
    "LOG_FILE":       "upload.log",
    "PROGRESS_FILE":  "upload_progress.json",
    "TEST_RUN":       False,
    "WORKERS":        8,
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("roboflow_uploader")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(CONFIG["LOG_FILE"], encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logger()


def load_progress() -> set:
    path = Path(CONFIG["PROGRESS_FILE"])
    if path.exists():
        data = json.loads(path.read_text())
        done = set(data.get("uploaded", []))
        log.info(f"Retomando upload — {len(done):,} imagens já enviadas anteriormente.")
        return done
    return set()


def save_progress(uploaded: set) -> None:
    Path(CONFIG["PROGRESS_FILE"]).write_text(
        json.dumps({"uploaded": list(uploaded), "last_save": datetime.now().isoformat()}, indent=2)
    )


def collect_images() -> list[dict]:
    items = []
    for split, base_dir in CONFIG["PATHS"].items():
        if not base_dir or not base_dir.exists():
            log.warning(f"[{split}] caminho não encontrado: {base_dir}")
            continue

        images_dir = base_dir / "images"
        labels_dir = base_dir / "labels"

        if not images_dir.exists():
            log.warning(f"[{split}] pasta 'images' não encontrada: {images_dir}")
            continue

        if not labels_dir.exists():
            log.warning(f"[{split}] pasta 'labels' não encontrada — imagens sem anotação.")

        imgs = [f for f in images_dir.iterdir() if f.suffix.lower() in CONFIG["IMG_EXTS"]]

        sem_label = 0
        for img in imgs:
            ann = labels_dir / img.with_suffix(".txt").name
            if not ann.exists():
                sem_label += 1
                ann = None
            items.append({
                "image_path":      str(img),
                "annotation_path": str(ann) if ann else None,
                "split":           split,
            })

        msg = f"[{split:>5}]  {len(imgs):>6,} imagens encontradas."
        if sem_label:
            msg += f"  (⚠ {sem_label} sem label)"
        log.info(msg)

    return items


_lock = threading.Lock()


def _upload_one(project, item: dict, uploaded: set, counters: dict, bar: tqdm) -> None:
    img  = item["image_path"]
    ann  = item["annotation_path"]
    splt = item["split"]

    try:
        project.single_upload(
            image_path=img,
            annotation_path=ann,
            split=splt,
            is_prediction=False,
            num_retry_uploads=2,
        )
        with _lock:
            uploaded.add(img)
            counters["ok"] += 1
            if counters["ok"] % CONFIG["BATCH_SIZE"] == 0:
                save_progress(uploaded)
        log.debug(f"OK  {img}")

    except Exception as exc:
        with _lock:
            counters["err"] += 1
        log.error(f"ERRO [{img}]: {exc}")

    finally:
        bar.update(1)
        with _lock:
            bar.set_postfix(erros=counters["err"], refresh=False)


def upload_dataset(project, items: list[dict], uploaded: set) -> None:
    pending  = [it for it in items if it["image_path"] not in uploaded]
    total    = len(items)
    already  = total - len(pending)
    counters = {"ok": 0, "err": 0}

    log.info("─" * 60)
    log.info(f"Total     : {total:,} imagens")
    log.info(f"Já feitas : {already:,}")
    log.info(f"Pendentes : {len(pending):,}")
    log.info(f"Workers   : {CONFIG['WORKERS']} threads paralelas")
    log.info("─" * 60)

    if not pending:
        log.info("Nada a enviar. Dataset já está completo.")
        return

    work = pending[:1] if CONFIG["TEST_RUN"] else pending
    if CONFIG["TEST_RUN"]:
        log.info("TEST_RUN ativo — enviando apenas 1 imagem.")

    bar = tqdm(
        total=len(work) + already,
        initial=already,
        desc="Enviando",
        unit="img",
        dynamic_ncols=True,
        colour="cyan",
    )

    with ThreadPoolExecutor(max_workers=CONFIG["WORKERS"]) as pool:
        futures = {pool.submit(_upload_one, project, item, uploaded, counters, bar): item for item in work}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                log.error(f"Thread encerrou com exceção: {exc}")

    bar.close()
    save_progress(uploaded)

    log.info("─" * 60)
    log.info(f"Concluído  — enviadas: {counters['ok']:,}  |  erros: {counters['err']:,}")
    if counters["err"]:
        log.info(f"Verifique '{CONFIG['LOG_FILE']}' para detalhes dos erros.")
    log.info("─" * 60)


def main() -> None:
    log.info("=" * 60)
    log.info("Roboflow Dataset Uploader")
    log.info(f"Workspace : {CONFIG['WORKSPACE']}")
    log.info(f"Project   : {CONFIG['PROJECT']}")
    for split, p in CONFIG["PATHS"].items():
        log.info(f"  [{split:>5}]  {p}")
    log.info("=" * 60)

    try:
        rf      = Roboflow(api_key=CONFIG["API_KEY"])
        ws      = rf.workspace(CONFIG["WORKSPACE"])
        project = ws.project(CONFIG["PROJECT"])
        log.info("Conexão com Roboflow: OK")
    except Exception as exc:
        log.error(f"Falha ao conectar ao Roboflow: {exc}")
        raise SystemExit(1)

    items = collect_images()
    if not items:
        log.error("Nenhuma imagem encontrada. Verifique o caminho do dataset.")
        raise SystemExit(1)

    uploaded = load_progress()

    start = time.time()
    upload_dataset(project, items, uploaded)
    elapsed = time.time() - start

    mins, secs = divmod(int(elapsed), 60)
    log.info(f"Tempo total: {mins}m {secs}s")


if __name__ == "__main__":
    main()