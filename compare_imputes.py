# not up-to-date, look at Reconstruction Results notebook to see this code updated and in action
def load_image_geotiff(path):
    loaded = load_raster(path, crop=(224, 224))
    return (loaded[0:3],loaded)

def load_image_jpg(path):
    image_file = Image.open(path)
    image_file.load()
    loaded_image = np.asarray( image_file, dtype="uint16" )
    crop=(224, 224)
    loaded_image = loaded_image.transpose(2,0,1)
    loaded_image = loaded_image[:, -crop[0]:, -crop[1]:]

    loaded_image[0,:]  = loaded_image[0,:] * ((means_list[0] + std_list[0] * 2) / 256)
    loaded_image[1,:]  = loaded_image[1,:] * ((means_list[1] + std_list[1] * 2) / 256)
    loaded_image[2,:]  = loaded_image[2,:] * ((means_list[2] + std_list[2] * 2) / 256)
    return (loaded_image, None)

means = np.array(means_list).reshape(-1, 1, 1)
stds = np.array(std_list).reshape(-1, 1, 1)

def normalize(img,channels,duplicate = False):
    normalized = img.copy()
    normalized = ((img - means[0:channels]) / stds[0:channels])
    if duplicate:
        normalized = np.concatenate((normalized,normalized))
    normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

def process_image(input_image_tup):
    rgb,full = input_image_tup

    to_evaluate = [rgb]
    
    # 1. Fill in with normalized original image
    to_evaluate.append(normalize(rgb, 3,duplicate=True))

    # 2. Fill in with original image
    dup_filled = np.concatenate((rgb,rgb))
    to_evaluate.append(normalize(dup_filled, 6))

    # 3. Fill in with means (so zeros after standardization)
    mean_tensor = (np.ones([3, *rgb.shape[1:]]) * np.array(means_list)[len(rgb):, None, None])
    mean_filled = np.concatenate((rgb,mean_tensor))
    to_evaluate.append(normalize(mean_filled, 6))
    
    # 4. Use actual infrared bands if you have them
    if full:
        to_evaluate.append(normalize(full,6))
    return tuple(to_evaluate)

# Input: list of tuples(3-4) of images
def evaluate_images(image_tuple_lists,mask_ratio):
    results = []
    for img_n, img in enumerate(image_tuple_lists):
        with torch.no_grad():
            original = img[0]
            loss1, pred1, mask,noise = model(img[1], mask_ratio=mask_ratio)
            loc_results = [mask,original,img[0],(loss1,pred1)]
            for i in range(2,len(img)):
                print(img[i].shape)
                loc_results.append(model(img[i], mask_noise=noise)[0:1]) # HAVE TO EDIT Prithvi.py model code give a custom mask
            results.append(tuple(loc_results))

    return results


def plot_image_mask_reconstruction(result):
    mask, original_img, normalized, loss_preds = result[0],result[1],result[2],result[3:]
    print(mask)
    # Mix visible and predicted patches
    mask_img_np = mask.numpy().reshape(6, 224, 224).transpose((1, 2, 0))[..., :3]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6))

    for subplot in ax:
        subplot.axis('off')

    ax[0].imshow(enhance_raster_for_visualization(original_img))
    masked_img_np = enhance_raster_for_visualization(original_img).copy()
    masked_img_np[mask_img_np[..., 0] == 1] = 0
    ax[1].imshow(masked_img_np)

    for i in range(2,len(loss_preds)):
        loss,pred = loss_preds[i]
        rec_img = normalized.clone()
        rec_img[mask == 1] = pred[mask == 1]  # binary mask: 0 is keep, 1 is remove
        rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means

        ax[i].imshow(enhance_raster_for_visualization(rec_img_np, ref_img=original_img))
        ax[i].set_title(loss)
    
file_paths = ["ESP_070975_2060.tif"]
to_evaluate = []
for path in file_paths:
    if path.contains(".tif"):
        loaded = load_image_geotiff(path)
    else:
        loaded = load_image_jpg(path)

    # append tuple of 3-4 normalized inputs to the list
    to_evaluate.append(process_image(loaded))
results = evaluate_images(to_evaluate)

for result_object in results:
    plot_image_mask_reconstruction(result_object)

    