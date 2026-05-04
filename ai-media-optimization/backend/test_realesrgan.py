from backend.inference.realesrgan_service import RealESRGANService


def main():
    service = RealESRGANService()

    output = service.enhance_image(
        input_path="storage/test_inputs/test-cat.png",
        output_path="storage/test_outputs/test_cat_x4.png",
        scale=4,
    )

    print(f"Enhanced image saved at: {output}")


if __name__ == "__main__":
    main()