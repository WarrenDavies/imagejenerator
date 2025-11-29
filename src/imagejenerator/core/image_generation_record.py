from dataclasses import dataclass
import csv

@dataclass
class ImageGenerationRecord():
    output_file_path: str = ""
    filename: str = ""
    timestamp: str = ""
    model: str = ""
    prompt: str = ""
    seed: str = ""
    height: int = 0
    width: int = 0
    inf_steps: int = 0
    guidance_scale: float = 0
    total_generation_time: float = 0
    generation_time_per_image: float = 0
    image_rating: float = -1


    def append_to_csv(self, path_to_csv):
        with open(path_to_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.filename,
                self.timestamp,
                self.model,
                self.prompt,
                self.seed,
                self.height,
                self.width,
                self.inf_steps,
                self.guidance_scale,
                self.total_generation_time,
                self.generation_time_per_image,
                self.image_rating,
            ])
