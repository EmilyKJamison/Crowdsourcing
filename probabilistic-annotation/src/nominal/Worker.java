package nominal;

public class Worker
{
    public int id;
    public Double[] bias; //added to label to produce latent gold
    public Double[] reliability; 
    public Double[][] labels;
    
    public Worker(int aId){
        id = aId;
    }
    
    public int getId(){
        return id;
    }
    public void setId(int id){
        this.id = id;
    }
    public Double[] getBias(){
        return bias;
    }
    public void setBias(Double[] bias){
        this.bias = bias;
    }
    public Double[] getReliability(){
        return reliability;
    }
    public void setReliability(Double[] reliability){
        this.reliability = reliability;
    }
    public Double[][] getLabels(){
        return labels;
    }
    public void setLabels(Double[][] labels){
        this.labels = labels;
    }
}
