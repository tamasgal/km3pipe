#!groovy
// DOCKER_IMAGES = ["python:3.5.0", "python:3.5.5", "python:3.6.4", "python:3.6.5"]
DOCKER_IMAGES = ["python:3.5.0"]
CHAT_CHANNEL = '#km3pipe'
DEVELOPERS = ['tgal@km3net.de', 'mlotze@km3net.de']

properties([gitLabConnection('KM3NeT GitLab')])


def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside("-u root:root") {

            // The following line causes a weird issue, where pip tries to 
            // install into /usr/local/... instead of the virtual env.
            // Any help figuring out what's happening is appreciated.
            //
            // def PYTHON_VENV = docker_image.replaceAll('[:.]', '') + 'venv'
            //
            // So we set it to 'venv' for all parallel builds now
            def DOCKER_NAME = docker_image.replaceAll('[:.]', '')
            def DOCKER_HOME = env.WORKSPACE + '/' + DOCKER_NAME + '_home'
            withEnv(["HOME=${env.WORKSPACE}", "MPLBACKEND=agg", "DOCKER_NAME=${DOCKER_NAME}"]){
                gitlabBuilds(builds: ["Install (${DOCKER_NAME})", "Test (${DOCKER_NAME})", "Docs (${DOCKER_NAME})"]) {
                    stage("Install (${DOCKER_NAME})") {
                        gitlabCommitStatus("Install (${DOCKER_NAME})") {
                            try { 
                                sh """
                                    pip install -U pip setuptools wheel
                                    make dependencies
                                    make install
                                """
                            } catch (e) { 
                                sendChatMessage("Install (${DOCKER_NAME}) Failed")
                                sendMail("Install (${DOCKER_NAME}) Failed")
                                throw e
                            }
                        }
                    }
                    stage("Test (${DOCKER_NAME})") {
                        gitlabCommitStatus("Test (${DOCKER_NAME})") {
                            try { 
                                sh """
                                    make clean
                                    make test
                                """
                            } catch (e) { 
                                sendChatMessage("Test Suite (${DOCKER_NAME}) Failed")
                                sendMail("Test Suite (${DOCKER_NAME}) Failed")
                                throw e
                            }
                            try { 
                                sh """
                                    make test-km3modules
                                """
                            } catch (e) { 
                                sendChatMessage("KM3Modules Test Suite (${DOCKER_NAME}) Failed")
                                sendMail("KM3Modules Test Suite (${DOCKER_NAME}) Failed")
                                throw e
                            }
                            try { 
                                step([$class: 'XUnitBuilder',
                                    thresholds: [
                                        [$class: 'SkippedThreshold', failureThreshold: '0'],
                                        [$class: 'FailedThreshold', failureThreshold: '0']],
                                    // thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
                                    tools: [[$class: 'JUnitType', pattern: 'reports/*.xml']]])
                            } catch (e) { 
                                sendChatMessage("Failed to create test reports (${DOCKER_NAME}).")
                                sendMail("Failed to create test reports (${DOCKER_NAME}).")
                                throw e
                            }
                            try { 
                                sh """
                                    make clean
                                    make test-cov
                                """
                                step([$class: 'CoberturaPublisher',
                                        autoUpdateHealth: false,
                                        autoUpdateStability: false,
                                        coberturaReportFile: "reports/coverage${DOCKER_NAME}.xml",
                                        failNoReports: false,
                                        failUnhealthy: false,
                                        failUnstable: false,
                                        maxNumberOfBuilds: 0,
                                        onlyStable: false,
                                        sourceEncoding: 'ASCII',
                                        zoomCoverageChart: false])
                            } catch (e) { 
                                sendChatMessage("Coverage (${DOCKER_NAME}) Failed")
                                sendMail("Coverage  (${DOCKER_NAME}) Failed")
                                throw e
                            }
                        }
                    }
                    stage("Docs (${DOCKER_NAME})") {
                        gitlabCommitStatus("Docs (${DOCKER_NAME})") {
                            try { 
                                sh """
                                    cd doc
                                    make html
                                """
                            } catch (e) { 
                                sendChatMessage("Building Docs (${DOCKER_NAME}) Failed")
                                sendMail("Building Docs (${DOCKER_NAME}) Failed")
                                throw e
                            }
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
