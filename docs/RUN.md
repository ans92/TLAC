# Running TLAC and SLAC Models

In order to run the model, first ensure that all the datasets are properly downloaded and configured as described in [DATASETS.md](https://github.com/ans92/TLAC/blob/main/docs/DATASETS.md). Once data is downloaded, paste the path of the dataset folder in ``` scripts/tlac/tlac_test.sh ```. After this you need to get the Gemini API key from Google Cloud Platform. You will get $300 credit once you signup for Google Cloud Platform. But in order to signup, you need an internationally recognized credit/debit card (or maybe other payment method is also possible). Once you have a Gemini API key, paste it also in the ``` scripts/tlac/tlac_test.sh ``` file. Now go to the main folder of project ```TLAC/``` and run the following command:

```
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat, imagenet_a, imagenet_r, imagenet_sketch, imagenetv2]

bash scripts/tlac/tlac_test.sh caltech101 SLAC
```
"SLAC" is the model name. If you want to run TLAC model then run the following command:

```
bash scripts/tlac/tlac_test.sh caltech101 TLAC
```
