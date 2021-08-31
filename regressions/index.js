require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear_regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
})
regression.train()
const r2 = regression.test(testFeatures, testLabels)
console.log('R2 is', r2)
plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'MSE'
})
regression.predict([
    [120, 380, 2],
]).print()
// console.log('Updated M is', regression.weigths.get(1, 0))
// console.log('Updated B is', regression.weigths.get(0, 0))