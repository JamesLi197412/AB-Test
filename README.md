###  Mobile Games: A/B Testing

A/B test is to be analyzed from the popular mobile puzzle game, Cookie Cats. The data is about A/B test with a mobile game, Cookie Cats. The data link is written [here](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing/data) , please have a look.

The purpose of this repo is to show my understandings on A/B test. It is one practical experience for A/B Test.

In the dataset, 
* two version/groups: gate_30 & gate_40, 
* measureable result: retention_1, retention_7 & sum_gamerounds

### A/B Testing Procedure
1. Understand the Problem
   1. If mobile game shall adjust its version from gate_30 to gate_40 so that the game rounds could increase and 1-day retention and 7-day retention could decrease.
   
2. Hypothesis Testing
   1. Set Up Hypothesis 
      1. Null Hypothesis (H0): There is no statistical difference between gate_30 && gate_40
      2. Alternative Hypothesis (Ha) : There is statistical difference between two groups.
      
   2. Analysis the result
       i) Because the data is given, choose the statistical method
      1. method 1:Mann Whitney U Test Result
      2. method 2: Bootstrapping multiple times and find difference two versions'median value. Generate the density plot to show 

3. Interpret results
    If the result is lower than threshold (p-value), then H0 hypothesis will be rejected, in other word, there is a statistically signficiant difference between them.
Else, there is no statistical difference between them.

### Conclusion

<img alt="Test Result" height="400" src="output/Test Result.png.png" width="350"/>

## Versioning

Github/Git are used for versioning/sharing. 

## Authors

* **James Li** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details