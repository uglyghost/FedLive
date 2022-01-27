accuracy(any(isnan(accuracy), 2),:) = [];
precision(any(isnan(precision), 2),:) = [];
predictedTile(any(isnan(predictedTile), 2),:) = [];
recall(any(isnan(recall), 2),:) = [];

aveACC_1 = mean(accuracy);
avePRE_1 = mean(precision);
aveREC_1 = mean(recall);
aveTIL_1 = mean(predictedTile);