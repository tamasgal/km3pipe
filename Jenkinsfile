pipeline {
    agent {
        docker { image 'python:3.6.5-stretch' }
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
