pipeline {
    agent {
        docker { image 'python:3.6.5' }
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
