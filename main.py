from typing import Union

from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.responses import FileResponse
from io import BytesIO
import cv2
import uuid
import os
import aiohttp
import numpy as np
from PIL import Image
from datetime import datetime
import rembg
from pathlib import Path

ORIGINAL ="original/"
MASK ="masked/"
OVERLAY="overlay_images/"
ORIGINAL = os.path.join(ORIGINAL)
if not os.path.exists(ORIGINAL):
    os.makedirs(ORIGINAL)

MASK = os.path.join(MASK)
if not os.path.exists(MASK):
    os.makedirs(MASK)

OVERLAY = os.path.join(OVERLAY)
if not os.path.exists(OVERLAY):
    os.makedirs(OVERLAY)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename =f"{timestamp}.png"
mask_filename=f"mask_BW_{filename}"
app = FastAPI()
async def download_image_and_process(image_url: str):
    #load image
    global mask_filename
    global filename
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url,timeout=300) as response:
            contents = await response.read()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"{timestamp}.png"
    with open(f"{ORIGINAL}{filename}", "wb") as f:
        f.write(contents)
    output_data = rembg.remove(contents)
    with open(f"{MASK}mask_{filename}", "wb") as overlay_file:
        overlay_file.write(output_data)

    mask_image=cv2.imread(f'{MASK}mask_{filename}',cv2.IMREAD_UNCHANGED)
    _,mask=cv2.threshold(mask_image[:,:,3],145,255,cv2.THRESH_BINARY)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
   

    mask_filename = f"mask_BW_{filename}"
    cv2.imwrite(f"{MASK}{mask_filename}",mask)
    
    mask=cv2.imread(f"{MASK}{mask_filename}",cv2.IMREAD_GRAYSCALE)
    image=cv2.imread(f"{ORIGINAL}{filename}",cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 3:  # If the image doesn't have an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    mask = cv2.bitwise_not(mask)
    overlay_mask = np.zeros_like(image)
    overlay_mask[:, :, 3] = mask
    overlay_mask[mask != 0] = [0, 0, 255, 128]
    cv2.imwrite(f"{MASK}mask_o_{filename}",overlay_mask)

    o_image=Image.open(f"{ORIGINAL}{filename}")
    color_mask_image=Image.open(f"{MASK}mask_o_{filename}")
    color_mask_image=color_mask_image.resize(o_image.size)
    merge_image=Image.alpha_composite(o_image.convert("RGBA"),color_mask_image.convert("RGBA"))
    merge_image.save(f"{OVERLAY}o_mask_{filename}")

    return {"filename": filename, "mask_filename": f"{MASK}{mask_filename}","output_file_path": f"mask_{filename}","overlay_mask":f"{OVERLAY}o_mask_{filename}"}

@app.post("/image_url/")
async def remove_background(image_url: str):
    filename = await download_image_and_process(image_url)
    return {"image": filename}

def get_image_path(filename: str,image_type:str="Mask") -> Path:
    
    if image_type=="Mask":
        return Path(MASK)/ f"mask_{filename}"
    elif image_type =="Mask B&W":
        return Path(MASK)/f"mask_BW_{filename}"
    else:
        raise ValueError(f"Invalid image type:{image_type}")
    
@app.get("/image/{filename}")
async def get_object_detect_image(filename: str,image_type:str=Query(...,description="Type of image to return",defaut="Mask",enum=["Mask","Mask B&W"])): 
    
    image_path = get_image_path(filename,image_type)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
    
    
def create_overlay_mask(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    gray_img = img.convert('L')
    # Convert the grayscale image to black and white
    bw_array = np.array(gray_img)
    threshold = 160  # Threshold for binarization
    bw_array[bw_array <= threshold] = 0
    bw_array[bw_array > threshold] =255
    # Convert the NumPy array back to a PIL image
    bw_img = Image.fromarray(bw_array.astype(np.uint8))
    return bw_img

@app.post("/merge_images/")

async def process_image(filename: str,file: UploadFile = File(...)  ):
    global mask_filename
    original_path = os.path.join(ORIGINAL, filename)
    original_image=cv2.imread(original_path)

    filename = file.filename

    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")  

    try:
        mask_image = create_overlay_mask(await file.read())
        # Save the mask image to a BytesIO object
        output_image = BytesIO()
        mask_image.save(output_image, format='PNG')
        output_image.seek(0)
        # Save the image to the folder
        with open(f'{OVERLAY}B&W_{filename}', 'wb') as f:
            f.write(output_image.read())

        mask_bw_image=cv2.imread(f"{MASK}{mask_filename}",cv2.IMREAD_GRAYSCALE)
        overlay_bw_image=cv2.imread(f"{OVERLAY}B&W_{filename}",cv2.IMREAD_GRAYSCALE)
        merged_mask = cv2.bitwise_or(mask_bw_image, overlay_bw_image)
        merged_mask_filename = f"{OVERLAY}merged_mask_{filename}"
        cv2.imwrite(merged_mask_filename, merged_mask)


        mask2=cv2.imread(f"{OVERLAY}merged_mask_{filename}",cv2.IMREAD_GRAYSCALE)
        original_image_rgb=cv2.cvtColor(original_image,cv2.COLOR_RGB2RGBA )
        merged_image = cv2.bitwise_or(original_image_rgb, original_image_rgb, mask=mask2)
        cv2.imwrite(f"{OVERLAY}merge_{filename}", merged_image)

        return {'file_path': str(f'{OVERLAY}B&W_{filename}'),'merge_image':f"{OVERLAY}merge_{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    

def get_mereg_image(filename: str,image_Type:str="Merge Image" ) -> Path:
    if  image_Type=="Merge Image":
        return Path(OVERLAY)/f"merge_o_mask_{filename}"
    elif image_Type=="Mask B&W":
        return Path(OVERLAY)/f"merged_mask_o_mask_{filename}"
    else:
        raise ValueError(f"Invalid image type:{image_Type}")

@app.get("/merge_image/{filename}")
async def get_image(filename: str,image_Type:str=Query(...,description="Type of image to return",defaut="Merge Image",enum=["Merge Image","Mask B&W"])): 
    
    image_path = get_mereg_image(filename,image_Type)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

    
