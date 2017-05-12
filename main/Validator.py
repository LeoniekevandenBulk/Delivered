# Class to evaluate Unet output on validation set with ground truth
class Validator:
    
    def __init__(self, batch_size, patch_size, group_labels, group_percentages):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.group_labels = group_labels
        self.group_percentages = group_percentages

    # Accumulate errors and thresholds over validation batches
    def get_validation(self, filenames, batchGenerator, network):
        
        sum_dice = 0
        sum_threshold = 0
        
        for filename in filenames:
            vol = "../data/volume-{0}".format(filename)
            vol_proxy = nib.load(vol)
            vol_array = vol_proxy.get_data()
            
            seg = "../data/segmentation-{0}".format(filename)
            seg_proxy = nib.load(seg)
            seg_array = seg_proxy.get_data()
            
            (batch, labels) = batchGenerator.get_batch(vol_array, seg_array, self.batch_size, 
                                     self.patch_size, self.group_labels, self.group_percentages)
            
            heatmaps = network.predict_fn(batch)
            
            error = get_error(heatmaps, labels)
            
            sum_dice += error[1]
            sum_threshold += error[2]
            
        return (sum_dice/len(filenames), sum_threshold/len(filenames))
         
    # Get error scores on Unet output (now dice only)
    def get_error(self, heatmaps, labels):
        
        # Determine best dice for the heatmap
        best_dice = 0 
        best_thres = 0
        for threshold in range(0,1,0.01):
            dice = dice_calc(heatmaps, labels, threshold)
            
            if(dice > best_dice):
                best_dice = dice
                best_thres = threshold
                
        return ("dice", best_dice, best_thres)
        
    # Determine dice score for a particular threshold    
    def dice_calc(self, heatmaps, labels, threshold):
        
        # Convert heatmap probabilities to binary prediction map via threshold value
        prediction = (heatmaps >= threshold).astype(int)
        
        # Calculate dice score
        dice = metric.dc(prediction, labels)
        
        return dice
