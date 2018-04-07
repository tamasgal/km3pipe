pipeline {
    agent {
        docker { image 'python:3.6.5-alpine3.6' }
    }
    stages {
        stage('Test') {
            steps {
                sh 'apk add git'
                sh 'pip install git+http://git.km3net.de/km3py/km3pipe@develop'
            }
        }
    }
}
