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

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
#  Ajuste ROOT_DIR para a pasta raiz do dataset
#  (onde ficam train2017/, valid2017/, test2017/ e annotations/)
# ─────────────────────────────────────────────
ROOT_DIR = Path(os.getenv("ROOT_DIR", ""))          # ex: C:/datasets/coco2017

CONFIG = {
    "API_KEY":   os.getenv("API_KEY"),
    "WORKSPACE": os.getenv("WORKSPACE"),
    "PROJECT":   os.getenv("PROJECT"),

    # Mapeamento split → (pasta de imagens, arquivo de anotação COCO)
    "SPLITS": {
        "train": {
            "images_dir":      ROOT_DIR / "train2017",
            "annotation_file": ROOT_DIR / "annotations" / "train_annotations.coco.json",
        },
        "valid": {
            "images_dir":      ROOT_DIR / "valid2017",
            "annotation_file": ROOT_DIR / "annotations" / "valid_annotations.coco.json",
        },
        "test": {
            "images_dir":      ROOT_DIR / "test2017",
            "annotation_file": ROOT_DIR / "annotations" / "test_annotations.coco.json",
        },
    },

    "IMG_EXTS":      {".jpg", ".jpeg", ".png"},
    "BATCH_SIZE":    50,
    "LOG_FILE":      "upload.log",
    "PROGRESS_FILE": "upload_progress.json",
    "TEST_RUN":      False,   # True → envia só 1 imagem (teste rápido)
    "WORKERS":       8,
}


# ─────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  PROGRESSO
# ─────────────────────────────────────────────
def load_progress() -> set:
    path = Path(CONFIG["PROGRESS_FILE"])
    if path.exists():
        data = json.loads(path.read_text())
        done = set(data.get("uploaded", []))
        log.info(f"Retomando upload — {len(done):,} imagens já enviadas.")
        return done
    return set()


def save_progress(uploaded: set) -> None:
    Path(CONFIG["PROGRESS_FILE"]).write_text(
        json.dumps(
            {"uploaded": list(uploaded), "last_save": datetime.now().isoformat()},
            indent=2,
        )
    )


# ─────────────────────────────────────────────
#  COLETA DE IMAGENS (formato COCO)
# ─────────────────────────────────────────────
def _build_ann_index(coco_data: dict) -> dict[int, list[dict]]:
    """Devolve {image_id: [anotações]} a partir do JSON COCO."""
    index: dict[int, list[dict]] = {}
    for ann in coco_data.get("annotations", []):
        index.setdefault(ann["image_id"], []).append(ann)
    return index


def _coco_ann_to_yolo(ann_list: list[dict], img_w: int, img_h: int) -> str:
    """
    Converte lista de anotações COCO (bbox = [x, y, w, h]) para YOLO TXT.
    Retorna string com uma linha por objeto.
    """
    lines = []
    for ann in ann_list:
        cat_id = ann.get("category_id", 0)
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw / 2) / img_w
        cy = (y + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h
        lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines)


def collect_images() -> list[dict]:
    """
    Lê cada arquivo .coco (JSON COCO) e cruza com as imagens presentes
    na pasta correspondente. Imagens referenciadas no JSON mas ausentes
    no disco são silenciosamente ignoradas.
    """
    items = []

    for split, cfg in CONFIG["SPLITS"].items():
        images_dir = cfg["images_dir"]
        ann_file   = cfg["annotation_file"]

        # ── Verifica pastas e arquivo de anotação ──────────────────────
        if not images_dir or not images_dir.exists():
            log.warning(f"[{split}] pasta de imagens não encontrada: {images_dir}")
            continue

        if not ann_file or not ann_file.exists():
            log.warning(f"[{split}] arquivo de anotação não encontrado: {ann_file}")
            continue

        # ── Carrega JSON COCO ──────────────────────────────────────────
        try:
            coco_data = json.loads(ann_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error(f"[{split}] falha ao ler {ann_file}: {exc}")
            continue

        ann_index  = _build_ann_index(coco_data)
        images_on_disk = {
            f.name: f
            for f in images_dir.iterdir()
            if f.suffix.lower() in CONFIG["IMG_EXTS"]
        }

        found = skipped = no_ann = 0

        for img_info in coco_data.get("images", []):
            filename = img_info.get("file_name", "")
            # Normaliza: o JSON pode guardar só o nome ou um sub-caminho
            fname = Path(filename).name

            if fname not in images_on_disk:
                skipped += 1
                log.debug(f"[{split}] imagem não encontrada no disco, ignorando: {fname}")
                continue

            img_path = images_on_disk[fname]
            img_id   = img_info["id"]
            img_w    = img_info.get("width",  0)
            img_h    = img_info.get("height", 0)

            anns = ann_index.get(img_id, [])
            if not anns:
                no_ann += 1

            # Converte anotações COCO → YOLO (string em memória)
            yolo_txt = _coco_ann_to_yolo(anns, img_w, img_h) if anns and img_w and img_h else None

            items.append({
                "image_path":  str(img_path),
                "yolo_labels": yolo_txt,   # None se sem anotação
                "split":       split,
            })
            found += 1

        msg = f"[{split:>5}]  {found:>6,} imagens encontradas."
        if skipped:
            msg += f"  ({skipped} no JSON mas ausentes no disco → ignoradas)"
        if no_ann:
            msg += f"  (⚠ {no_ann} sem anotação)"
        log.info(msg)

    return items


# ─────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────
_lock = threading.Lock()


def _upload_one(
    project,
    item: dict,
    uploaded: set,
    counters: dict,
    bar: tqdm,
    tmp_dir: Path,
) -> None:
    img  = item["image_path"]
    splt = item["split"]

    # Grava label YOLO em arquivo temporário (Roboflow aceita arquivo .txt)
    ann_path = None
    tmp_file = None
    if item["yolo_labels"] is not None:
        tmp_file = tmp_dir / (Path(img).stem + ".txt")
        tmp_file.write_text(item["yolo_labels"], encoding="utf-8")
        ann_path = str(tmp_file)

    try:
        project.single_upload(
            image_path=img,
            annotation_path=ann_path,
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
        if tmp_file and tmp_file.exists():
            tmp_file.unlink()   # limpa arquivo temporário
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

    # Pasta temporária para os labels YOLO gerados em memória
    tmp_dir = Path("_tmp_labels")
    tmp_dir.mkdir(exist_ok=True)

    bar = tqdm(
        total=len(work) + already,
        initial=already,
        desc="Enviando",
        unit="img",
        dynamic_ncols=True,
        colour="cyan",
    )

    with ThreadPoolExecutor(max_workers=CONFIG["WORKERS"]) as pool:
        futures = {
            pool.submit(_upload_one, project, item, uploaded, counters, bar, tmp_dir): item
            for item in work
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                log.error(f"Thread encerrou com exceção: {exc}")

    bar.close()
    save_progress(uploaded)

    # Remove pasta temporária se vazia
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    log.info("─" * 60)
    log.info(f"Concluído  — enviadas: {counters['ok']:,}  |  erros: {counters['err']:,}")
    if counters["err"]:
        log.info(f"Verifique '{CONFIG['LOG_FILE']}' para detalhes dos erros.")
    log.info("─" * 60)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("Roboflow Dataset Uploader  [formato COCO]")
    log.info(f"Workspace : {CONFIG['WORKSPACE']}")
    log.info(f"Project   : {CONFIG['PROJECT']}")
    for split, cfg in CONFIG["SPLITS"].items():
        log.info(f"  [{split:>5}]  imgs: {cfg['images_dir']}")
        log.info(f"           anns: {cfg['annotation_file']}")
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