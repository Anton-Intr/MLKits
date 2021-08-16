const outputs = []

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel])
  // console.log(outputs)
}

function runAnalysis() {
  // Write code here to analyze stuff
  const test_set_size = 100
  const [test_set, training_set] = split_dataset(min_max(outputs, 3), test_set_size)
  _.range(1, 20).forEach(k => {
    const accuracy = _.chain(test_set)
      .filter(test_point => knn(training_set, _.initial(test_point), k) === test_point[3])
      .size()
      .divide(test_set_size)
      .value()
    console.log('Accuracy: ', accuracy, ' for k: ', k)
  })
}

function knn(data, point, k) {
  return _.chain(data)
    .map(row => {
      return [
        distance(_.initial(row), point), _.last(row)
      ]
    })
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value()
}

function distance(pointA, pointB) {
  return _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => (a - b) ** 2)
    .sum()
    .value() ** 0.5
}

function split_dataset(data, testCount) {
  const shuffled = _.shuffle(data)
  const test_set = _.slice(shuffled, 0, testCount)
  const training_set = _.slice(shuffled, testCount)
  return [test_set, training_set]
}

function min_max(data, feature_count) {
  const cloned_data = _.cloneDeep(data)
  for (let i = 0; i < feature_count; i++) {
    const column = cloned_data.map(row => row[i])
    const min = _.min(column)
    const max = _.max(column)
    for (let j = 0; j < cloned_data.length; j++) {
      cloned_data[j][i] = (cloned_data[j][i] - min) / (max - min)
    }
  }
  return cloned_data
}

