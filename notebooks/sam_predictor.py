import torch, os, gc, cv2, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    # overlays annotations on the image
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def try_clean(items):
    # cleans up workspace completely after inference on an image/tile
    for i in items:
        try:
            del(i)
        except:
            print('unable to delete: {}'.format(i))
            pass
    gc.collect()
    torch.cuda.empty_cache()

# define parameters
annotate_boxes_on_image = False
min_area = 1000
ppside = 32
thresh_iou = 0.85
thresh_stability = 0.90
crop_n_layers = 1
crop_n_pts_df = 2
write_mask_objects = False
write_mask_objects_by_image = False

cwd = os.getcwd()
# specify the model
sam_checkpoint = os.path.join(cwd,"models/sam_vit_h_4b8939.pth")
model_type = "vit_h"

# use GPU for inference
device = "cuda"

in_rast = 'M:/My Drive/'
# input_dir = os.path.join('images','GLSC_test_images_unknown')
# input_dir = os.path.join('images','GLSC_images_for_presentations')
input_dir = 'M:/My Drive/FastSAM/images'
# out_dir = os.path.join(input_dir,'outputs_sam')
out_dir = os.path.join(os.path.split(input_dir)[0],'outputs_sam')
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# define and load the SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# initialize mask generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=ppside,
    pred_iou_thresh=thresh_iou,
    stability_score_thresh=thresh_stability,
    crop_n_layers=crop_n_layers,
    crop_n_points_downscale_factor=crop_n_pts_df,
    min_mask_region_area=min_area)

for filename in os.listdir(input_dir):
    f = os.path.join(input_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        # read image
        image = cv2.imread(os.path.join(input_dir,filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # create masks
        masks = mask_generator.generate(image)
        # plot figure
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(os.path.join(out_dir,filename.split('.')[0]+'_with_annotations.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print('Wrote out {}'.format(os.path.join(out_dir,filename.split('.')[0]+'_with_annotations.png')))
        
        if write_mask_objects_by_image:
            output_directory = os.path.join(out_dir,'objects',filename.split('.')[0])
        else:
            output_directory = os.path.join(out_dir,'objects')
            
        # if annotate_boxes_on_image:
        #     plt.figure(figsize=(20,20))
        #     for m in masks:
        #         if m['area'] > 100000:
        #             print(m['bbox'])
        #             # plot image with bounding boxes
        #             x,y,w,h = m['bbox']
        #             image = cv2.rectangle(img=image, rec=(x,y,w,h), color=(255,0,0), thickness=10)
        #             image = cv2.putText(image, text="IoU: "+str(m['predicted_iou']), org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0))
        #     plt.axis("off")
        #     plt.savefig(os.path.join(out_dir,filename.split('.')[0]+'_with_bbox_annotations.png'), bbox_inches='tight', dpi=300)
        #     plt.close()
        #     print('Wrote out {}'.format(os.path.join(out_dir,filename.split('.')[0]+'_with_bbox_annotations.png')))

        if write_mask_objects:
            if write_mask_objects_by_image:
                output_directory = os.path.join(out_dir,'objects',filename.split('.')[0])
            else:
                output_directory = os.path.join(out_dir,'objects')
            # create the output directory
            if not os.path.isdir(output_directory):
                os.mkdir(output_directory)
            for m in masks:
                if not os.path.isdir(os.path.join(out_dir,'objects',filename.split('.')[0])):
                    os.mkdir(os.path.join(out_dir,'objects',filename.split('.')[0]))
                x,y,w,h = m['bbox']
                ROI = image[y:y+h, x:x+w]
                outfile = os.path.join(output_directory,filename.split('.')[0]+'_'+str(x)+'_'+str(y)+'_'+str(w)+'_'+str(h)+'.png')
                cv2.imwrite(outfile, cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))
                del(outfile)

        # clean up workspace and free-up memory
        try_clean([image,sam,mask_generator,masks])