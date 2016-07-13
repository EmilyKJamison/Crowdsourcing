package numeric;


public class EstimatorNum
{
    /*
     * Let's assume 6 annotator and 5 total documents, with at least 3 annotations per doc.  
     * Annotations are in the range 1-3.
     */
    Double[] annotator1 = {1.0, 2.0, null, null, 1.0};
    Double[] annotator2 = {null, 2.0, 2.0, null, 1.0};
    Double[] annotator3 = {1.0, 1.0, 2.0, 3.0, null};
    Double[] annotator4 = {2.0, null, 3.0, 3.0, null};
    Double[] annotator5 = {null, null, null, 3.0, 1.0};
    Double[] annotator6 = {null, null, null, 3.0, null};
    
    Double[][] originalDataset = {annotator1, annotator2, annotator3, 
                                    annotator4, annotator5, annotator6};
    Double[][] updatedDataset;
    Double[] predictedGolds;
    Double[] previousPredictedGolds; //to watch for convergence
    Double[] workerNoise;
    int iterationCounter;
    
    public static final boolean USENOISEWEIGHTING = true;
    
    // set initial variables
    public EstimatorNum(){
        updatedDataset = originalDataset;
        iterationCounter = 0;
        predictedGolds = new Double[5];
        previousPredictedGolds = new Double[5];
        workerNoise = new Double[6];
        for (int i=0; i<workerNoise.length; i++){
            workerNoise[i] = 0.0;
        }
        for (int i=0; i<previousPredictedGolds.length; i++){
            previousPredictedGolds[i] = 0.0;
        }
    }
    
    // calculate the new average across labels
    public void expectationStep(){
        iterationCounter++;
        for(int docId=0;docId<5;docId++){
            int numLabels = 0;
            Double labelSum = 0.0;
            Double reliabilitySum = 0.0;
            for(int annotator=0; annotator< updatedDataset.length; annotator++){
                Double label = updatedDataset[annotator][docId];
                if (label != null){
                    numLabels++;
                    Double workerReliability = 1 / (workerNoise[annotator] + 0.1);
                    reliabilitySum = reliabilitySum + workerReliability;
                    if(USENOISEWEIGHTING){
                        labelSum = labelSum + (label * workerReliability);
                    }else{
                        labelSum = labelSum + label;
                    }
                }
            }
            
            Double newPredictedGold;
            if(USENOISEWEIGHTING){
                newPredictedGold = labelSum / reliabilitySum; //original
//                newPredictedGold = (labelSum / (reliabilitySum / numLabels) / numLabels); // rewritten to match EstimatorNom, but equivalent
            }else{
                newPredictedGold = labelSum / numLabels; // i.e. the avg label
            }
            if(iterationCounter > 1){
                previousPredictedGolds[docId] = predictedGolds[docId];
            }
            predictedGolds[docId] = newPredictedGold;
        }
    }
    
    // update the labels based on bias distance from the new avgs
    public void maximizationStep(){
        for(int annotator=0;annotator<updatedDataset.length;annotator++){
            
            // Find a worker's bias and noisiness
            Double biasSum = 0.0;
//            Double noiseSum = 0.0;//to be cut from first loop
            int numAnnotations = 0;
            for(int anno=0; anno<updatedDataset[annotator].length;anno++){
                Double label = updatedDataset[annotator][anno];
                if (label != null){
                    numAnnotations++;
                    Double biasOfALabel = label - predictedGolds[anno];
                    biasSum = biasSum + biasOfALabel;
//                    noiseSum = noiseSum + Math.abs(biasOfALabel);//to be cut from first loop
                }
                
            }
            Double aWorkerBias = biasSum / numAnnotations;
//            Double aWorkerNoise = noiseSum / numAnnotations; //to be cut from first loop
            
            // Update the worker's labels to account for the newfound bias
            for(int anno=0; anno<updatedDataset[annotator].length;anno++){
                Double label = updatedDataset[annotator][anno];
                if (label != null){
                    updatedDataset[annotator][anno] = label - aWorkerBias;
                }
            }
            //make another loop here 
            // Find a worker's bias and noisiness
            Double noiseSum = 0.0;
            numAnnotations = 0;
            for(int anno=0; anno<updatedDataset[annotator].length;anno++){
                Double label = updatedDataset[annotator][anno];
                if (label != null){
                    numAnnotations++;
                    Double biasOfALabel = label - predictedGolds[anno];
                    noiseSum = noiseSum + Math.abs(biasOfALabel);
                }
                
            }
            Double aWorkerNoise = noiseSum / numAnnotations; 
            
            
            
            
            // Make note of the noise
            workerNoise[annotator] = aWorkerNoise;
        }
    }
    public void watchProgress(){
        
        System.out.println("Iteration " + iterationCounter);
        boolean stillConverging = false;
        for(int doc=0; doc<previousPredictedGolds.length; doc++){
            Double dif = previousPredictedGolds[doc] - predictedGolds[doc];
            System.out.println("Doc " + doc + " dif: " + dif);
            if(dif != 0.0){
                stillConverging = true;
            }
        }
        // converges at 43 without noise weighting
        // converges at 243 with current noise weighting
        if(!stillConverging){
            System.out.println();
            for(int doc=0; doc<predictedGolds.length; doc++){
                System.out.println("Doc " + doc + ": finalGold: " + predictedGolds[doc]);
                
            }
        }
    }

    public static void main(String[] args)
    {
        EstimatorNum emNumeric = new EstimatorNum();
        for(int i=0;i<243;i++){
            emNumeric.expectationStep();
            emNumeric.maximizationStep();
            emNumeric.watchProgress();
        }

    }

}
