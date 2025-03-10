# Initial Predictions

This folder where you get initial predictions from a Pretrained language model , this folder is structured as follow :

```
Initial_Predictions
 |-- Dataset
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- retacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- Models
 |    |-- tacred
 |    |    |-- RoBERTa_TACRED.bin        
 |    |-- retacred
 |    |    |-- RoBERTa_ReTACRED.bin
 |-- Predictions
 |    |-- Initial_predictions.csv          
```


where :

- **Models**: Contains pretrained language models (PLMs) fine-tuned on relation extraction (RE) datasets, used for generating predictions.  
- **Dataset**: Includes the **TACRED** and **ReTACRED** datasets.  
- **Predictions**: Stores the initial predictions obtained from the PLM on the test set or any other sample data.  


 # How to get initial predictions 

To generate initial predictions on a test set or any sample data with the same structure, use `Get_Predictions.py` as follows:  

 ```bash
python3 Get_Predictions.py \
    --model_name_or_path "roberta-large" \
    --check_model "./Models/retacred/RoBERTa_ReTACRED.bin" \
    --test_path "./Dataset/retacred/test.json" \
    --predictions_path "./Predictions/Initial_predictions.csv" \
    --dataset_type "RETACRED"

``` 
