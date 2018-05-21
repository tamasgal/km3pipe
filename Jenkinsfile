#!groovy
// def DOCKER_IMAGES = ["python:3.5.5", "python:3.6.4", "python:3.6.5"]
DOCKER_IMAGES = ["python:3.6.5"]
CHAT_CHANNEL = '#km3pipe'
DEVELOPERS = ['tgal@km3net.de', 'mlotze@km3net.de']

properties([gitLabConnection('KM3NeT GitLab')])


def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside {

            // The following line causes a weird issue, where pip tries to 
            // install into /usr/local/... instead of the virtual env.
            // Any help figuring out what's happening is appreciated.
            //
            // def PYTHON_VENV = docker_image.replaceAll('[:.]', '') + 'venv'
            //
            // So we set it to 'venv' for all parallel builds now
            def DOCKER_HOME = env.WORKSPACE + '/' + docker_image.replaceAll('[:.]', '') + '_home'
            def PYTHON_VENV = DOCKER_HOME + '/venv'
            withEnv(["HOME=${env.WORKSPACE}"]){
                stage("${docker_image}") {
                    echo "Running in ${docker_image} with HOME set to ${DOCKER_HOME}"
                }
                stage("Prepare") {
                    sh "rm -rf ${PYTHON_VENV}"
                    sh "python -m venv ${PYTHON_VENV}"
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        pip install -U pip setuptools wheel
                    """
                }
                gitlabBuilds(builds: ['Deps', 'Test', 'Install', 'Test KM3Modules', 'Test Reports', 'Coverage', 'Docs']) {
                    stage("Deps") {
                        gitlabCommitStatus("Deps") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    make dependencies
                                """
                            } catch (e) { 
                                sendChatMessage("Install Dependencies Failed")
                                sendMail("Install Dependencies Failed")
                                throw e
                            }
                        }
                    }
                    stage('Test') {
                        gitlabCommitStatus("Test") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    make clean
                                    make test
                                """
                            } catch (e) { 
                                sendChatMessage("Test Suite Failed")
                                sendMail("Test Suite Failed")
                                throw e
                            }
                        }
                    }
                    stage("Install") {
                        gitlabCommitStatus("Install") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    make install
                                """
                            } catch (e) { 
                                sendChatMessage("Install Failed")
                                sendMail("Install Failed")
                                throw e
                            }
                        }
                    }
                    stage('Test KM3Modules') {
                        gitlabCommitStatus("Test KM3Modules") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    make test-km3modules
                                """
                            } catch (e) { 
                                sendChatMessage("KM3Modules Test Suite Failed")
                                sendMail("KM3Modules Test Suite Failed")
                                throw e
                            }
                        }
                    }
                    stage('Test Reports') {
                        gitlabCommitStatus("Test Reports") {
                            try { 
                                step([$class: 'XUnitBuilder',
                                    thresholds: [
                                        [$class: 'SkippedThreshold', failureThreshold: '0'],
                                        [$class: 'FailedThreshold', failureThreshold: '0']],
                                    // thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
                                    tools: [[$class: 'JUnitType', pattern: 'reports/*.xml']]])
                            } catch (e) { 
                                sendChatMessage("Failed to create test reports.")
                                sendMail("Failed to create test reports.")
                                throw e
                            }
                        }
                    }
                    stage('Coverage') {
                        gitlabCommitStatus("Coverage") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    make clean
                                    make test-cov
                                """
                                step([$class: 'CoberturaPublisher',
                                        autoUpdateHealth: false,
                                        autoUpdateStability: false,
                                        coberturaReportFile: 'reports/coverage.xml',
                                        failNoReports: false,
                                        failUnhealthy: false,
                                        failUnstable: false,
                                        maxNumberOfBuilds: 0,
                                        onlyStable: false,
                                        sourceEncoding: 'ASCII',
                                        zoomCoverageChart: false])
                                publishHTML target: [
                                   allowMissing: false,
                                   alwaysLinkToLastBuild: false,
                                   keepAll: true,
                                   reportDir: 'reports/coverage',
                                   reportFiles: 'index.html',
                                   reportName: 'Coverage'
                                ]
                            } catch (e) { 
                                sendChatMessage("Coverage Failed")
                                sendMail("Coverage Failed")
                                throw e
                            }
                        }
                    }
                    stage('Docs') {
                        gitlabCommitStatus("Docs") {
                            try { 
                                sh """
                                    . ${PYTHON_VENV}/bin/activate
                                    cd doc
                                    export MPLBACKEND="agg"
                                    make html
                                """
                            } catch (e) { 
                                sendChatMessage("Building Docs Failed")
                                sendMail("Building Docs Failed")
                                throw e
                            }
                        }
                    }
                }
                stage('Publishing Docs') {
                    try {
                       publishHTML target: [
                           allowMissing: false,
                           alwaysLinkToLastBuild: false,
                           keepAll: true,
                           reportDir: 'doc/_build/html',
                           reportFiles: 'index.html',
                           reportName: 'Documentation'
                       ]
                    } catch (e) {
                        sendChatMessage("Publishing Docs Failed")
                        sendMail("Publishing Docs Failed")
                    }
                }
            }
        }
    }
    return stages
}


node('master') {

    cleanWs()
    checkout scm

    def stages = [:]
    for (int i = 0; i < DOCKER_IMAGES.size(); i++) {
        def docker_image = DOCKER_IMAGES[i]
        stages[docker_image] = get_stages(docker_image)
    }

    parallel stages
}


def sendChatMessage(message, channel=CHAT_CHANNEL) {
    rocketSend channel: channel, message: "${message} - [Build ${env.BUILD_NUMBER} ](${env.BUILD_URL})"
}


def sendMail(subject, message='', developers=DEVELOPERS) {
    for (int i = 0; i < developers.size(); i++) {
        def developer = DEVELOPERS[i]
        emailext (
            subject: "$subject - Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
            body: """
                <p>$message</p>
                <p>Check console output at <a href ='${env.BUILD_URL}'>${env.BUILD_URL}</a> to view the results.</p>
            """,
            to: developer
        )    
    }
}
