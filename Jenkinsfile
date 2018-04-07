pipeline {
    agent {
        docker { image 'python:3.6.5-alpine3.6' }
    }
    stages {
        stage('Build') {
            steps {
                sh 'make install-dev'
            }
        }
        stage('Test') {
            steps {
                sh 'make test'
            }
        }
    }
}
