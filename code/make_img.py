from PIL import Image, ImageDraw, ImageFont
import os


def combine_images_with_titles(image_paths, grid_size, titles, font_path, font_size=40, padding=20, save_path=None):
    # Open all images and store them in a list
    images = [Image.open(img_path) for img_path in image_paths]

    # Get the size of the individual images
    img_width, img_height = images[0].size

    # Calculate the height of a single row and column
    title_height = font_size  # This can be adjusted based on the font size

    # Adjust combined image size to include padding around the grid
    combined_width = img_width * grid_size[1] + padding * (grid_size[1] + 1)  # Padding on both sides
    combined_height = img_height * grid_size[0] + title_height + padding * (
                grid_size[0] + 1) +20 # Padding on both sides and space for titles

    # Create a new image with the appropriate size for the grid
    new_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

    # Create a drawing context to add titles
    draw = ImageDraw.Draw(new_image)

    # Load a custom TrueType font (make sure the path to the font file is correct)
    font = ImageFont.truetype(font_path, font_size)  # Adjust font size here

    # Paste the images into the grid with padding
    for i, img in enumerate(images):
        row = i // grid_size[1]
        col = i % grid_size[1]

        # Calculate the position to paste the image, accounting for padding
        image_position = (
            (col + 1) * padding + col * img_width,  # Add padding around the image
            (row + 1) * padding + row * img_height  # Add padding around the image
        )
        new_image.paste(img, image_position)

        # Add the title only for the last row
        if row == grid_size[0] - 1:  # Check if it's the last row
            # Get the bounding box of the text (x, y, width, height)
            bbox = draw.textbbox((0, 0), titles[col], font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate the position to center the text under the image
            title_position = (
                (col + 1) * padding + col * img_width + (img_width - text_width) // 2,  # Center the text
                (row + 1) * padding + row * img_height + img_height + 5  # Adjust text position below the image
            )

            # Draw the title
            draw.text(title_position, titles[col], fill="black", font=font)

    # Save the combined image if save_path is provided
    if save_path:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the image
        new_image.save(save_path)
        print(f"Image saved at {save_path}")
    else:
        # Return the combined image for further use (if no save_path provided)
        return new_image

# List of image file paths (replace with your actual image paths)

image_paths = [
   '/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_2600/粘贴的图像 (7).png',
    '/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_3200/粘贴的图像 (7).png',
    '/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_5500/粘贴的图像 (7).png',
    '/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/HSMI_7800/粘贴的图像 (7).png',
]

#image_paths = [
#   '/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_02/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A1E9Q1_00_03/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_A2N8V0_00_05/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/MnMs_D4N6W6_00_08/粘贴的图像 (7).png',
#]

#image_paths = [
#   '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient007_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient011_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient025_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_patient040_efficient_unet.png',
#]

#image_paths = [
#   '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A1E9Q1_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_A2N8V0_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_D4N6W6_efficient_unet.png',
#    '/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_gt.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_ours.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_unet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_swinunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_mambaunet.png','/home/colin/Mamba-UNet-main/可视化/cam可视化/grad_cam_E0O0S0_efficient_unet.png',
#]

#image_paths = [
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/CAMUS_70_gt.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_070_2CH_ED/粘贴的图像 (6).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_255_2CH_ED/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_349_4CH_ES/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/CAMUS_432_4CH_ES/粘贴的图像 (7).png'
#]

#image_paths = [
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_007_01/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_011_02/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_075_01/粘贴的图像 (7).png',
#    '/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (2).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像.png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (3).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (4).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (5).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (6).png','/home/colin/Mamba-UNet-main/可视化/分割可视化/ACDC_093_01 /粘贴的图像 (7).png'
#]

# Set the grid size (number of rows, number of columns)
grid_size = (4, 7)  # Adjust this based on your layout

# Titles for the columns (this should match the number of columns in your grid)
titles = ['Image', 'GT', 'Ours', 'U-Net', 'Swin-UNet', 'Mamba-UNet', 'Efficient-UNet']


# Specify the path to the .ttf font file (e.g., Arial, or a custom font)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'  # Modify to your font path

# Combine the images with titles
save_directory = '/home/colin/Mamba-UNet-main/可视化'
save_file_name = 'HSMI可视化.png'
save_path = os.path.join(save_directory, save_file_name)

# Combine the images with titles and save to the specified path
combine_images_with_titles(image_paths, grid_size, titles, font_path, font_size=40, save_path=save_path)

