import cv2
from pathlib import Path


def images_to_video(images_dir: Path, video_id: str, output_path: Path, fps: float = 30.0):
    # Procura todas as imagens do vídeo específico (ex: video09_*.jpg)
    image_paths = sorted(images_dir.glob(f"{video_id}_*.jpg"))
    if not image_paths:
        print(f"Nenhuma imagem encontrada para {video_id} em {images_dir}")
        return

    # Lê a primeira imagem para descobrir a resolução (largura e altura)
    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        print(f"Erro ao ler a primeira imagem: {image_paths[0]}")
        using_placeholder = True
        return

    height, width = first_frame.shape[:2]

    # Configura o VideoWriter (codec mp4v para gerar .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Convertendo {len(image_paths)} frames para {output_path.name}...")
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Vídeo salvo com sucesso em: {output_path}")


if __name__ == "__main__":
    # Ajuste o caminho do seu projeto conforme necessário
    project_root = Path(__file__).resolve().parents[2] if "__file__" in locals() else Path(".")
    images_dir = project_root / "datasets" / "anti_uav_tracking_test_final" / "images" / "test"
    output_dir = project_root / "videos_exportados"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Converte o vídeo 9 e o vídeo 10
    for v_id in ["video16", "video19"]:
        output_file = output_dir / f"{v_id}.mp4"
        images_to_video(images_dir, v_id, output_file, fps=30.0)