

# kNN

Необходимо построить правило классификации кварталов.
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
# Регрессионный анализ

Наша задача спрогнозировать запасы сгущенного молока.
В первую очередь посмотрим на график

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-1-1.png)<!-- -->

### Ответим на 4 вопроса:
#### 1. Тренд есть и он линейный.
#### 2. Сезонность наблюдается, причем она аддитивная, нет ярко выраженного "веера".
#### 3. В середине графика наблюдается скачок, но в целом кажется, что ряд убывает.
#### 4. Наблюдается подозрительное значение (32 наблюдение). Уберем его.


```r
condmilk[32] = (condmilk[31] + condmilk[33])/2
```


### Построим регрессионную модель.



В качестве базового месяца берем январь.


```r
res = lm(condmilk ~ time +                  month.02 + month.03 + month.04 + 
           month.05 + month.06 + month.07 + month.08 + month.09 + month.10 + 
           month.11 + month.12, milk)

summary(res)
```

```
## 
## Call:
## lm(formula = condmilk ~ time + month.02 + month.03 + month.04 + 
##     month.05 + month.06 + month.07 + month.08 + month.09 + month.10 + 
##     month.11 + month.12, data = milk)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -46.328  -7.185   1.661   7.323  29.436 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 70.51244    5.23915  13.459  < 2e-16 ***
## time        -0.09992    0.03985  -2.507 0.013678 *  
## month.02    -4.51808    6.72974  -0.671 0.503437    
## month.03    -6.90317    6.73009  -1.026 0.307339    
## month.04     3.11875    6.73068   0.463 0.644045    
## month.05    32.70767    6.73151   4.859 4.06e-06 ***
## month.06    54.65359    6.73257   8.118 8.70e-13 ***
## month.07    72.74950    6.73387  10.804  < 2e-16 ***
## month.08    75.05242    6.73540  11.143  < 2e-16 ***
## month.09    67.90134    6.73717  10.079  < 2e-16 ***
## month.10    52.55025    6.73917   7.798 4.42e-12 ***
## month.11    26.19617    6.74141   3.886 0.000177 ***
## month.12     7.57109    6.74388   1.123 0.264095    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 15.05 on 107 degrees of freedom
##   (12 observations deleted due to missingness)
## Multiple R-squared:  0.8207,	Adjusted R-squared:  0.8006 
## F-statistic: 40.82 on 12 and 107 DF,  p-value: < 2.2e-16
```

Коэффициент **R-squared2** не такой большой, как бы нам хотелось. Что ж, посмотрим на график.


```r
pred  = predict.lm(res, milk)

plot(pred, type = "l", col = "red", ylim = c(40, 160))
lines(condmilk, col = "green")
```

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-5-1.png)<!-- -->

Хм, наша модель плохо описывает первую половину наших данных(что ожидаемо). Попробуем построить наш прогноз по второй половине, т.к. характер ряда во второй половине много стабильней, чем в первой (может быть в начале путь фирма пыталась наладить свое производство, поэтому мы видим такие скачки, но за последние 5 лет ситуация стабилизировалась)


```r
milk.2 = milk[60:132, ]

res.2 = lm(condmilk ~ time +                month.02 + month.03 + month.04 + 
           month.05 + month.06 + month.07 + month.08 + month.09 + month.10 + 
           month.11 + month.12, milk.2)

summary(res.2)
```

```
## 
## Call:
## lm(formula = condmilk ~ time + month.02 + month.03 + month.04 + 
##     month.05 + month.06 + month.07 + month.08 + month.09 + month.10 + 
##     month.11 + month.12, data = milk.2)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -22.5175  -5.6537   0.4443   5.4295  19.5845 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  71.2161     8.5180   8.361 6.31e-11 ***
## time         -0.1042     0.0816  -1.277 0.207796    
## month.02     -3.0178     6.9932  -0.432 0.668013    
## month.03     -2.5036     6.9947  -0.358 0.721963    
## month.04      9.4486     6.9970   1.350 0.183231    
## month.05     34.3848     7.0004   4.912 1.09e-05 ***
## month.06     55.6409     7.0046   7.943 2.68e-10 ***
## month.07     70.0571     7.0099   9.994 2.58e-13 ***
## month.08     76.9433     7.0160  10.967 1.13e-14 ***
## month.09     72.1955     7.0232  10.280 1.02e-13 ***
## month.10     56.9757     7.0312   8.103 1.54e-10 ***
## month.11     27.9279     7.0402   3.967 0.000242 ***
## month.12      5.3709     6.7075   0.801 0.427228    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 11.06 on 48 degrees of freedom
##   (12 observations deleted due to missingness)
## Multiple R-squared:  0.9032,	Adjusted R-squared:  0.879 
## F-statistic: 37.33 on 12 and 48 DF,  p-value: < 2.2e-16
```

**R-squared2** сильно вырос. Теперь мы замечаем, что время незначимый параметр, интерпретация проста: ситуация стабилизировалась. Но на графике все же виден небольшой спад. Сравним два прогноза.


```r
pred.1  = predict.lm(res, milk)
pred.2  = predict.lm(res.2, milk.2)

plot(pred.1[60:132], type = "l", col = "red", ylim = c(40, 160))
lines(condmilk[60:132], col = "green")
lines(pred.2, col = "blue")
```

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-7-1.png)<!-- -->

Видно, что второй прогноз не привнес много пользы. Откат модели.

#Анализ остатков.


```r
plot(res)
```

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-8-1.png)<!-- -->![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-8-2.png)<!-- -->![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-8-3.png)<!-- -->![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-8-4.png)<!-- -->

Судя по третьему графику разброс остатков нас устраивает, все остатки более менее распределены по всей площади графика.
А вот с нормальностью есть небольшие проблемы. Посмотрим на гистограму.


```r
hist(rstudent(res), main = "Распределение остатков", xlab = "residuals", breaks = 20)
```

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-9-1.png)<!-- -->

Немного завышен левый конец распределения и совсем отсутствует правый. Но в целом данное распределение не сильно отличается от нормального (есть некоторая симметрия и колоколобразность).

По четвертому графику можно сказать, что сильных выбросов нет.

### Распределение остатков по сезонам


```r
seasons = rep(1:12, 10)[1:120]

boxplot(rstudent(res) ~ seasons)
```

![alter text](https://github.com/xaphoon/some_tasks_on_R/blob/master/unnamed-chunk-10.png)<!-- -->

Сильно выделяющегося сезона не наблюдается.

### Автокорреляция остатков


```r
#H0: there is no correlation among residuals, i.e., they are independent.
#H1: residuals are autocorrelated.
library(lmtest)
```

```
## Loading required package: zoo
```

```
## 
## Attaching package: 'zoo'
```

```
## The following objects are masked from 'package:base':
## 
##     as.Date, as.Date.numeric
```

```r
dwtest(res)
```

```
## 
## 	Durbin-Watson test
## 
## data:  res
## DW = 0.35245, p-value < 2.2e-16
## alternative hypothesis: true autocorrelation is greater than 0
```

Наблюдается автокорреляция остатков, что ожидаемо от временных рядов.

# Объединение предикторов

Сразу бросается в глаза различие коэффициентов сезонов декабрь-апрель и май-октябрь. Плохо интерпретируются такие группы.
Попробуем сначала создать предиктор зимы.



Далее возьмем за базовый месяц август.


```
## 
## Call:
## lm(formula = condmilk ~ time + winter + month.03 + month.04 + 
##     month.05 + month.06 + month.07 + month.09 + month.10 + month.11, 
##     data = milk)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -46.300  -7.686   1.533   8.388  29.687 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 145.27692    5.39035  26.951  < 2e-16 ***
## time         -0.09527    0.03998  -2.383  0.01892 *  
## winter      -74.02082    5.52823 -13.390  < 2e-16 ***
## month.03    -81.93236    6.77203 -12.099  < 2e-16 ***
## month.04    -71.91509    6.77097 -10.621  < 2e-16 ***
## month.05    -42.33082    6.77014  -6.253 8.06e-09 ***
## month.06    -20.38955    6.76955  -3.012  0.00323 ** 
## month.07     -2.29827    6.76920  -0.340  0.73487    
## month.09     -7.15573    6.76920  -1.057  0.29280    
## month.10    -22.51145    6.76955  -3.325  0.00120 ** 
## month.11    -48.87018    6.77014  -7.218 7.44e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 15.14 on 109 degrees of freedom
##   (12 observations deleted due to missingness)
## Multiple R-squared:  0.8152,	Adjusted R-squared:  0.7983 
## F-statistic:  48.1 on 10 and 109 DF,  p-value: < 2.2e-16
```

Можно объединить июль-август-сентябрь. Назовем этот сезон полулето.



Теперь допустим базовым месяцем апрель.


```
## 
## Call:
## lm(formula = condmilk ~ time + winter + month.03 + month.05 + 
##     month.06 + semi.summer + month.10 + month.11, data = milk)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -46.302  -7.973   1.510   8.396  31.528 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  73.38148    5.29858  13.849  < 2e-16 ***
## time         -0.09561    0.03983  -2.400 0.018039 *  
## winter       -2.10539    5.50625  -0.382 0.702923    
## month.03    -10.01761    6.74369  -1.485 0.140252    
## month.05     29.58461    6.74369   4.387 2.63e-05 ***
## month.06     51.52622    6.74404   7.640 8.26e-12 ***
## semi.summer  68.76511    5.50841  12.484  < 2e-16 ***
## month.10     49.40567    6.74781   7.322 4.14e-11 ***
## month.11     23.04728    6.74934   3.415 0.000892 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 15.08 on 111 degrees of freedom
##   (12 observations deleted due to missingness)
## Multiple R-squared:  0.8133,	Adjusted R-squared:  0.7998 
## F-statistic: 60.43 on 8 and 111 DF,  p-value: < 2.2e-16
```

Объединим март-апрель. Теперь изначальная первая группа теперь состоит только из 2-х сезонов.



Возьмем за базовый месяц май.


```
## 
## Call:
## lm(formula = condmilk ~ time + winter + mar.apr + month.06 + 
##     semi.summer + month.10 + month.11, data = milk)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -46.300  -8.199   1.427   9.549  31.547 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 102.94548    5.34461  19.262  < 2e-16 ***
## time         -0.09526    0.04004  -2.379  0.01906 *  
## winter      -31.69000    5.53569  -5.725 8.80e-08 ***
## mar.apr     -34.59289    5.87179  -5.891 4.10e-08 ***
## month.06     21.94126    6.77992   3.236  0.00159 ** 
## semi.summer  39.17945    5.53699   7.076 1.37e-10 ***
## month.10     19.81931    6.78276   2.922  0.00421 ** 
## month.11     -6.53943    6.78406  -0.964  0.33715    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 15.16 on 112 degrees of freedom
##   (12 observations deleted due to missingness)
## Multiple R-squared:  0.8096,	Adjusted R-squared:  0.7977 
## F-statistic: 68.01 on 7 and 112 DF,  p-value: < 2.2e-16
```

Видим, что можно май объединить с ноябрем. Но смысла в этом мало. Оставим нашу модель как есть.

# Прогноз на 8 месяцев

```
     1       2      3       4       5       6        7        8      
---  ------  -----  ------  ------  ------  -------  -------  -------
     58.42   53.8   51.32   61.24   90.73   112.58   130.57   132.78 
```



