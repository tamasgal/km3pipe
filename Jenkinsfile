#!groovy
// def docker_images = ["python:3.5.5", "python:3.6.4", "python:3.6.5"]
def docker_images = ["python:3.6.5"]

def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside {

            // def PYTHON_VENV = docker_image.replaceAll('[:.]', '') + 'venv'
            def PYTHON_VENV = 'venv'
            def CHAT_CHANNEL = '#km3pipe'

            stage("${docker_image}") {
                echo "Running in ${docker_image}"
            }
            stage("Prepare") {
                sh "rm -rf ${PYTHON_VENV}"
                sh "python -m venv ${PYTHON_VENV}"
                sh """
                    . ${PYTHON_VENV}/bin/activate
                    pip install -U pip setuptools wheel
                """
            }
            stage("Build") {
                sendChatMessage(CHAT_CHANNEL, "Build Started")
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Deps") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make dependencies
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Install Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Doc Deps") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make doc-dependencies
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Install Doc Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Dev Deps") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make dev-dependencies
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Install Dev Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make clean
                        make test
                    """
                    junit 'junit.xml'
                    archive 'junit.xml'
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make install
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Install Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test KM3Modules') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make test-km3modules
                    """
                    junit 'junit_km3modules.xml'
                    archive 'junit_km3modules.xml'
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "KM3Modules Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Coverage') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make clean
                        make test-cov
                    """
                    step([$class: 'CoberturaPublisher',
                            autoUpdateHealth: false,
                            autoUpdateStability: false,
                            coberturaReportFile: 'coverage.xml',
                            failNoReports: false,
                            failUnhealthy: false,
                            failUnstable: false,
                            maxNumberOfBuilds: 0,
                            onlyStable: false,
                            sourceEncoding: 'ASCII',
                            zoomCoverageChart: false])
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Coverage Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Docs') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make doc-dependencies
                        cd docs
                        export MPLBACKEND="agg"
                        make html
                    """
                } catch (e) { 
                    rocketSend channel: CHAT_CHANNEL, message: "Building the Docs Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Publishing Docs') {
               publishHTML target: [
                   allowMissing: false,
                   alwaysLinkToLastBuild: false,
                   keepAll: true,
                   reportDir: 'docs/_build/html',
                   reportFiles: 'index.html',
                   reportName: 'Documentation'
               ]
            }


        }
    }
    return stages
}

node('master') {

    checkout scm

    def stages = [:]
    for (int i = 0; i < docker_images.size(); i++) {
        def docker_image = docker_images[i]
        stages[docker_image] = get_stages(docker_image)
        /* get_stages(docker_image) */
    }

    parallel stages

}


def sendChatMessage(channel, message) {
    rocketSend channel: channel, message: "${message} - [Build #${env.BUILD_NUMBER}](${env.BUILD_URL})"
}
