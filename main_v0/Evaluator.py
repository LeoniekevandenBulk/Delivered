from medpy import metric
from sklearn.metrics import log_loss

# Class to evaluate Unet output on validation set with ground truth
class Evaluator:

    # Accumulate errors and thresholds over validation batches
    def get_evaluation(self, predicted, targets):

        sum_dice = 0
        sum_threshold = 0
        sum_crossentropy = 0

        #Calculate error scores for every image from batch
        for pred,target in zip(predicted,targets):

            # Calculate dice score
            dice_error = get_dice_error(pred,target)
            sum_dice += dice_error[1]
            sum_threshold += dice_error[2]

            # Calculate cross-entropy
            sum_crossentropy += log_loss(target,pred)

        return ([("dice", sum_dice/predicted.shape[0], sum_threshold/predicted.shape[0]),("cross_entropy", sum_crossentropy/predicted.shape[0])])
         
    # Get dice error scores
    def get_dice_error(self, pred, target):
        
        # Determine best dice for the prediction
        best_dice = 0 
        best_thres = 0
        for threshold in range(0,1,0.01):
            
            # Convert heatmap probabilities to binary prediction map via threshold value
            prediction = (pred >= threshold).astype(int)
        
            # Calculate dice score
            dice = metric.dc(prediction, target)
            
            if(dice > best_dice):
                best_dice = dice
                best_thres = threshold
                
        return ("dice", best_dice, best_thres)
