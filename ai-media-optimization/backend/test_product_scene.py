from backend.inference.background_removal_service import BackgroundRemovalService
from backend.inference.product_scene_compositor import ProductSceneCompositor


def main():
    input_image = "storage/test_inputs/test-product.png"

    foreground_path = "storage/test_outputs/product_foreground.png"
    mask_path = "storage/test_outputs/product_mask.png"

    bg_service = BackgroundRemovalService()
    compositor = ProductSceneCompositor()

    bg_service.extract_foreground(
        input_path=input_image,
        output_path=foreground_path,
        mask_output_path=mask_path,
    )

    for bg in [
        "studio",
        "marble_table",
        "luxury_bathroom",
        "nature_background",
    ]:
        result = compositor.compose(
            foreground_path=foreground_path,
            background_key=bg,
            output_path=f"storage/test_outputs/product_{bg}.jpg",
        )

        print(result)


if __name__ == "__main__":
    main()