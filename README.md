# Задачи на R
Казаков Данил  

Необходимо построить правило классификации кварталов.

# kNN

Сначала произведем нормализацию признаков, т.к. наш метод зависит от расстояний между объектов.



```r
scaled.df = scale(df)
```

Смоделируем классификацию. Кварталы классифицируются на две группы:  "богатые" (для которых значение переменной "цена" выше медианы), и "бедные" (для которых значение переменной "цена" ниже медианы).


```r
cl = rep(0, nrow(scaled.df))
m = median(scaled.df[,14])

for (i in 1:nrow(scaled.df)){
  if (scaled.df[i, 14] >= m){
    cl[i] = 1
    }
}
```

Разделим выборку на **test** и **train**.


```r
set.seed(1234)
test.num = sample(1:nrow(scaled.df), 169, replace = FALSE)

test = scaled.df[test.num, -c(1:4, 9, 14)]
train = scaled.df[-test.num, -c(1:4, 9, 14)]
```

Подберем значение **k** перебором.


```r
a = matrix(rep(0,15), nrow = 1)
colnames(a) = c(1:15)
rownames(a) = c("sum. error")

for (i in 1:15){
  zzz = knn(train, test, cl[-test.num], k = i)
  a[i] = sum(zzz != cl[test.num])
}
kable(a, align = "l")
```
```
             1    2    3    4    5    6    7    8    9    10   11   12   13   14   15 
-----------  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---
sum. error   29   23   22   22   22   23   22   24   23   24   23   22   22   25   26 
```
Вполне целесообразно остановиться на значении **k = 3**.
Построим прогноз и таблицу сопряженности.


```r
zzz = knn(train, test, cl[-test.num], k = 3)
table(zzz, cl[test.num])
```

```
##    
## zzz  0  1
##   0 83 14
##   1  8 64
```

#### Вывод
В процентном соотношении мы имеем **13 %** ошибок. Поэтому мы можем говорить, что модель вполне рабочая.

# Classification tree

Разделим выборку на **test** и **train**.




```r
test = df[test.num, -c(1:4, 9, 14)]
train = df[-test.num, -c(1:4, 9, 14)]
```

Попробуем подобрать параметры нашей модели. В качестве параметра качества будем использовать **%** ошибок.

```
        (5; 2; 6)   (10; 5; 4)   (10; 5; 3)   (10; 5; 2)   (20; 10; 4) 
------  ----------  -----------  -----------  -----------  ------------
train   0.09        0.10         0.11         0.13         0.12        
test    0.15        0.15         0.16         0.14         0.15        
```

Следуя правилу KISS, остановимся на параметрах **(10; 5; 2)**


```r
res = rpart(cl ~ ., data = train, method = "class",
            control = rpart.control(minsplit = 10, minbucket = 5, maxdepth = 2))

rpart.plot(res, type = 2, extra = 1)
```

![alt text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-10-1.png)<!-- -->

```r
table(train$cl, predict(res, train[ , -9], type = "class"))
```

```
##    
##       0   1
##   0 152   8
##   1  36 141
```

```r
table(test$cl, predict(res, test[ , -9], type = "class"))
```

```
##    
##      0  1
##   0 87  4
##   1 20 58
```

#### Вывод
В процентном соотношении мы имеем 14% ошибок. Данная модель очень проста и это её +: мы используем всего 2 параметра (2 неравенства).

# Random Forest

Снова выделим **train** и **test**.

```r
df = Boston
test = df[test.num, -c(1:4, 9, 14)]
train = df[-test.num, -c(1:4, 9, 14)]

cl.1 = as.factor(cl)
```

Запустим процедуру **randomForest** с **ntree = 200** и попробуем найти оптимальное значение параметра ntree.


```r
set.seed(1234)
res = randomForest(train, y = cl.1[-test.num], ntree = 200, mtry = floor(sqrt(ncol(train))),
                       replace = FALSE, nodesize = 1,
                       importance = TRUE, localImp = FALSE,
                       proximity = FALSE, norm.votes = TRUE, do.trace = 10,
                       keep.forest = T, corr.bias = FALSE, keep.inbag = FALSE)
```

```
## ntree      OOB      1      2
##    10:  17.26% 18.24% 16.38%
##    20:  14.84% 15.00% 14.69%
##    30:  14.24% 13.75% 14.69%
##    40:  15.13% 14.37% 15.82%
##    50:  13.95% 12.50% 15.25%
##    60:  13.65% 12.50% 14.69%
##    70:  14.84% 13.12% 16.38%
##    80:  13.65% 10.62% 16.38%
##    90:  14.84% 13.12% 16.38%
##   100:  15.13% 13.75% 16.38%
##   110:  15.13% 14.37% 15.82%
##   120:  14.84% 13.75% 15.82%
##   130:  14.54% 13.12% 15.82%
##   140:  14.24% 12.50% 15.82%
##   150:  14.24% 12.50% 15.82%
##   160:  13.95% 12.50% 15.25%
##   170:  14.24% 11.88% 16.38%
##   180:  14.24% 11.88% 16.38%
##   190:  14.24% 11.88% 16.38%
##   200:  13.65% 10.62% 16.38%
```

По моему мнению, оптимальным значением количества деревьев явлется **60**.


```r
set.seed(1234)
res = randomForest(train, y = cl.1[-test.num], ntree = 60, mtry = floor(sqrt(ncol(train))),
                   replace = FALSE, nodesize = 1,
                   importance = TRUE, localImp = FALSE,
                   proximity = FALSE, norm.votes = TRUE, do.trace = 70,
                   keep.forest = T, corr.bias = FALSE, keep.inbag = FALSE)

table(cl.1[-test.num], predict(res, train))
```

```
##    
##       0   1
##   0 160   0
##   1   0 177
```

```r
round(sum(cl[-test.num] != predict(res, train))/nrow(train), 2)
```

```
## [1] 0
```

```r
table(cl.1[test.num], predict(res, test))
```

```
##    
##      0  1
##   0 83  8
##   1 11 67
```

```r
round(sum(cl[test.num] != predict(res, test))/nrow(test), 2)
```

```
## [1] 0.11
```

Как мы видим, на **train** выборке наш лес не ошибается, но на тестовой выборке мы имеем **11%** ошибок. Можно подумать, что здесь имеет место переобучение. Попробуем уменьшить число деревьев и увеличить **nodesize**.


```r
set.seed(1234)
res = randomForest(train, y = cl.1[-test.num], ntree = 20, mtry = floor(sqrt(ncol(train))),
                   replace = FALSE, nodesize = 10,
                   importance = TRUE, localImp = FALSE,
                   proximity = FALSE, norm.votes = TRUE, do.trace = 41,
                   keep.forest = T, corr.bias = FALSE, keep.inbag = FALSE)

table(cl.1[-test.num], predict(res, train))
```

```
##    
##       0   1
##   0 158   2
##   1  19 158
```

```r
round(sum(cl[-test.num] != predict(res, train))/nrow(train), 2)
```

```
## [1] 0.06
```

```r
table(cl.1[test.num], predict(res, test))
```

```
##    
##      0  1
##   0 86  5
##   1 13 65
```

```r
round(sum(cl.1[test.num] != predict(res, test))/nrow(test), 2)
```

```
## [1] 0.1
```

Методом перебора мы добились **9%** ошибок, что неплохо.

#### Информативность переменных.


```r
varImpPlot(res, sort = F)
```

![alt text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-15-1.png)<!-- -->

# GBM





```r
set.seed(1234)
gbm.res = gbm(cl ~. , data = train, distribution = "gaussian", n.trees = 200,
               shrinkage = 0.05, interaction.depth = 3, bag.fraction = 0.66,
               n.minobsinnode = 10, cv.folds = 0, keep.data = T, verbose = T)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        0.2350             nan     0.0500    0.0153
##      2        0.2218             nan     0.0500    0.0121
##      3        0.2101             nan     0.0500    0.0118
##      4        0.1987             nan     0.0500    0.0100
##      5        0.1890             nan     0.0500    0.0094
##      6        0.1801             nan     0.0500    0.0084
##      7        0.1721             nan     0.0500    0.0071
##      8        0.1638             nan     0.0500    0.0061
##      9        0.1567             nan     0.0500    0.0065
##     10        0.1508             nan     0.0500    0.0043
##     20        0.1119             nan     0.0500    0.0013
##     40        0.0820             nan     0.0500    0.0004
##     60        0.0697             nan     0.0500   -0.0002
##     80        0.0630             nan     0.0500   -0.0004
##    100        0.0580             nan     0.0500   -0.0001
##    120        0.0535             nan     0.0500   -0.0002
##    140        0.0502             nan     0.0500   -0.0001
##    160        0.0472             nan     0.0500   -0.0002
##    180        0.0447             nan     0.0500   -0.0003
##    200        0.0423             nan     0.0500   -0.0001
```

Разумно остановиться на **n.trees = 80**. Построим на результаты.

* #### train ####


```r
pr = predict(gbm.res, newdata = train[, -c(9)], n.trees = 80)
pr.2 = rep(0, nrow(train))
pr.2[(pr > 0.6)] = 1
table(cl[-test.num], pr.2)
```

```
##    pr.2
##       0   1
##   0 154   6
##   1  31 146
```

```r
round(sum(cl[-test.num] != pr.2)/nrow(train), 2)
```

```
## [1] 0.11
```

* #### test ####


```r
pr = predict(gbm.res, newdata = test[, -c(9)], n.trees = 80)
pr.2 = rep(0, nrow(test))
pr.2[(pr > 0.6)] = 1
table(cl[test.num], pr.2)
```

```
##    pr.2
##      0  1
##   0 88  3
##   1 18 60
```

```r
round(sum(cl[test.num] != pr.2)/nrow(test), 2)
```

```
## [1] 0.12
```

