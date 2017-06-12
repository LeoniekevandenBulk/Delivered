from medpy import metric
from sklearn.metrics import log_loss

'''
Evaluator
'''

# Class to evaluate Unet output on validation set with ground truth
class Evaluator:

    # Accumulate errors and thresholds over validation batches
    def get_evaluation(self, targets, predicted):

        sum_dice = 0
        sum_threshold = 0
        sum_crossentropy = 0

        #Calculate error scores for every image from batch
        #for pred,target in zip(predicted,targets):
        for i in range(predicted.shape[0]):
            pred = predicted[i, 0, :, :]
            target = targets[i, 0, :, :]

            # Calculate dice score
            dice_error = self.get_dice_error(pred,target)
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
        for threshold in [float(x)/100 for x in range(101)]:
            
            # Convert heatmap probabilities to binary prediction map via threshold value
            prediction = (pred >= threshold).astype(int)
        
            # Calculate dice score
            if target.min() == 0 and target.max() == 0 and prediction.min() == 0 and prediction.max() == 0:
                dice = 1.0
            else:
                dice = metric.dc(prediction, target)
            
            if(dice > best_dice):
                best_dice = dice
                best_thres = threshold
                
        return ("dice", best_dice, best_thres)
