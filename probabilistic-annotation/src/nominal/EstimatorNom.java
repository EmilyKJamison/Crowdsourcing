package nominal;

import java.util.ArrayList;
import java.util.List;

public class EstimatorNom
{
    // If wanted, add annotators and/or items, but not categories
    String[] annotator1 = {"apple",  "banana", null,     null,     "apple"};// item1, item2, ...
    String[] annotator2 = {null,     "banana", "banana", null,     "apple"};
    String[] annotator3 = {"apple",  "apple",  "banana", "orange", "apple"};
    String[] annotator4 = {"banana", "orange", "apple",  "orange", null};
    String[] annotator5 = {"apple",  "banana", null,     "orange", "apple"};
    String[] annotator6 = {"orange", "orange", "orange", "orange", "orange"};
    
    String[][] originalDataset = {annotator1, annotator2, annotator3, 
            annotator4, annotator5, annotator6};
    List<Worker> workers;
    List<String> categories; //currently hardcoded at 3
    Double[][] predictedGolds;
    Double[][] lastPredictedGolds;
    int iterationCounter;
    
    public static final boolean VERBOSE = true;
    public static final boolean USERELIABILITY = true;
    
    public EstimatorNom(){
        tabulateCategories();
        workers = new ArrayList<Worker>();
        for(int i=0;i<originalDataset.length;i++){
            Worker worker = new Worker(i);
            Double[] bias = {0.0, 0.0, 0.0};
            Double[] reliability = {1.0, 1.0, 1.0};
            worker.setLabels(categoryStringsToDoubles(originalDataset[i]));
            worker.setBias(bias);
            worker.setReliability(reliability);
            workers.add(worker);
        }
        predictedGolds = new Double[annotator1.length][];
        lastPredictedGolds = new Double[annotator1.length][];
        for(int i=0;i<annotator1.length;i++){
            Double[] prediction = {0.0, 0.0, 0.0};
            predictedGolds[i] = prediction;
            lastPredictedGolds[i] = new Double[3];
        }
        iterationCounter = 0;
    }
    private void tabulateCategories(){
        categories = new ArrayList<String>();
        for(String[] worker:originalDataset){
            for(String label:worker){
                if(!categories.contains(label) && label != null){
                    categories.add(label);
                    if(VERBOSE){
                        System.out.println("ln50 Adding category: " + label);
                    }
                }
            }
        }
    }
    private Double[][] categoryStringsToDoubles(String[] labels){
        Double[][] labelsDoubles = new Double[labels.length][];
        for(int item=0;item<labelsDoubles.length;item++){
            Double[] newLabel = new Double[categories.size()];
            for(int i=0;i<newLabel.length;i++){
                if(labels[item] == null){
                    newLabel[i] = null;
                }else if(labels[item].equals(categories.get(i))){
                    newLabel[i] = 1.0;
                }else{
                    newLabel[i] = 0.0;
                }
            }
            labelsDoubles[item] = newLabel;
        }
        return labelsDoubles;
    }
    public void expectationStep(){
        for(int i=0;i<predictedGolds.length;i++){
            Double[] sumArray = {0.0, 0.0, 0.0};
            Double[] sumReliabilities = {0.0, 0.0, 0.0};
            Integer[] sumLabels = {0, 0, 0};
            for(Worker worker: workers){
                Double[] originalLabel = worker.getLabels()[i];
                Double[] biases = worker.getBias();
                Double[] reliabilities = worker.getReliability();
                for(int c=0;c<sumArray.length;c++){
                    if(originalLabel[c] == null){
                        continue;
                    }
                    Double toBeAddedToSum = (originalLabel[c] + biases[c]) * ((1 / reliabilities[c]) );
                    if(!USERELIABILITY){
                        toBeAddedToSum = originalLabel[c] + biases[c];
                    }
                    sumReliabilities[c] = sumReliabilities[c] + (1 / reliabilities[c]);
                    sumLabels[c]++;
                    sumArray[c] = sumArray[c] + toBeAddedToSum;
                }
            }
            for(int c=0;c<sumArray.length;c++){
                lastPredictedGolds[i][c] = predictedGolds[i][c].doubleValue();
                
                if(!USERELIABILITY){
                    predictedGolds[i][c] = sumArray[c] / sumLabels[c];
                }else{
                    // TODO Normalization needs work/checking
                    // This might be correct, but weights are normalized by entire corpus, not per item.
                    predictedGolds[i][c] = (sumArray[c] / (sumReliabilities[c] / sumLabels[c])) / sumLabels[c]; 
                    // can be rewritten like this, which is the original version from EstimatorNum
//                    predictedGolds[i][c] = (sumArray[c] / sumReliabilities[c]);  
                    
                }
            }
        }
        iterationCounter++;
    }
    public void maximizationStep(){
        for(int w=0;w<workers.size();w++){
            Worker worker = workers.get(w);
            Double[] newBias = new Double[categories.size()];
            Double[] newReliability = new Double[categories.size()];
            for(int i=0;i<categories.size();i++){
                Double biasSum = 0.0;
                Double reliabilitySum = 0.0;
                for(int item=0; item<predictedGolds.length;item++){
                    Double labelPerCat = worker.getLabels()[item][i];
                    Double predictionPerCat = predictedGolds[item][i];
                    if(labelPerCat == null){
                        continue;
                    }
                    // Label is subtracted here and bias is added to label 
                    // in E-step when calculating new gold.
                    biasSum = biasSum + (predictionPerCat - labelPerCat);
                    reliabilitySum = reliabilitySum + Math.abs(predictionPerCat - labelPerCat);
                }
                newBias[i] = biasSum / predictedGolds.length;
                newReliability[i] = reliabilitySum / predictedGolds.length;
            }
            worker.setBias(newBias);
            worker.setReliability(newReliability);
            workers.set(w, worker);
        }
    }
    private Double r8(Double value){
        return Math.round(value * 10000000d) / 10000000d;
    }
    private Double r3(Double value){
        return Math.round(value * 100d) / 100d;
    }
    
    public boolean stop(){
        boolean stop = true;
        for(int i=0; i<predictedGolds.length;i++){
            for(int j=0; j<predictedGolds[i].length;j++){
                if(!r8(predictedGolds[i][j]).equals(r8(lastPredictedGolds[i][j]))){
                    stop = false;
                }
            }
        }
        return stop;
    }
    protected void printProgress(){
        for(int i=0; i<predictedGolds.length;i++){
            if(VERBOSE){
                System.out.println("ln104 iter " + iterationCounter 
                        + "  item " + i 
                        + "  " + Math.abs(lastPredictedGolds[i][0] - predictedGolds[i][0]) 
                        + "  " + Math.abs(lastPredictedGolds[i][1] - predictedGolds[i][1] )
                        + "  " + Math.abs(lastPredictedGolds[i][2] - predictedGolds[i][2]));
            }
        }
        if(VERBOSE){
            System.out.println("ln160 Warning: Convergence uses a little rounding.");
        }
        
    }
    protected void printFinal(){
        System.out.println("\n Final Results:");
        if(VERBOSE){
            for(Worker worker: workers){
                System.out.print("\nWorker " + worker.getId() + " bias: ");
                Double[] bias = worker.getBias();
                for(int a=0;a<bias.length;a++){
                    System.out.print(" " + r3(bias[a]));
                }
            }
            for(Worker worker: workers){
                System.out.print("\nWorker " + worker.getId() + " reliability: ");
                Double[] reliability = worker.getReliability();
                for(int a=0;a<reliability.length;a++){
                    System.out.print(" " + r3(reliability[a]));
                }
            }
            System.out.println();
        }
        for(int i=0;i<predictedGolds.length;i++){
            Double class0 = predictedGolds[i][0];
            Double class1 = predictedGolds[i][1];
            Double class2 = predictedGolds[i][2];
            String newGoldClass = "";
            if(class0 > class1 && class0 > class2){
                newGoldClass = categories.get(0);
            }else if(class1 > class0 && class1 > class2){
                newGoldClass = categories.get(1);
            }else if(class2 > class0 && class2 > class1){
                newGoldClass = categories.get(2);
            }
            if(VERBOSE){
                System.out.println("item " + i 
                        + "  " + r3(class0) 
                        + "  " + r3(class1) 
                        + "  " + r3(class2) 
                        + "  " + newGoldClass);
            }else{
                System.out.println("item " + i 
                        + "  " + newGoldClass);
            }
        }
        System.out.println("Total iterations: " + iterationCounter);
    }
    public void run(){
        expectationStep();
        maximizationStep();
        while(!stop()){
//        while(i<50){
            expectationStep();
            maximizationStep();
            printProgress();
        }
        printFinal();
    }
    public static void main(String[] args){
        EstimatorNom estimatorNom = new EstimatorNom();
        estimatorNom.run();
        
        System.out.println("Done!");
    }
}
