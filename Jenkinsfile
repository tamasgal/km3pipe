#!groovy
import groovy.io.FileType
import static groovy.io.FileType.FILES

DOCKER_FILES_DIR = './dockerfiles'
CHAT_CHANNEL = '#km3pipe-dev'
DEVELOPERS = ['tgal@km3net.de', 'mlotze@km3net.de']
MAIN_DOCKER = 'py365'

properties([gitLabConnection('KM3NeT GitLab')])




node('master') {

    cleanWs()
    checkout scm

    def customImage = ''
    dir("${DOCKER_FILES_DIR}"){
        customImage = docker.build("km3pipe:${env.BUILD_ID}", "-f ${MAIN_DOCKER} .")
    }

    customImage.inside("-u root:root") {
            withEnv(["HOME=${env.WORKSPACE}", "MPLBACKEND=agg"]){
                gitlabBuilds(builds: ["Install", "Test", "Docs"]) {

                        gitlabCommitStatus("Install") {
                            try { 
                                sh """
                                    pip install -U pip setuptools wheel
                                    make dependencies
                                    make install
                                """
                            } catch (e) { 
                                sendChatMessage("Install Failed")
                                sendMail("Install Failed")
                                throw e
                            }
                        }

                        gitlabCommitStatus("Test") {
                            try { 
                                sh """
                                    make clean
                                    make test
                                """
                            } catch (e) { 
                                sendChatMessage("Test Suite Failed")
                                sendMail("Test Suite Failed")
                                throw e
                            }
                            try { 
                                sh """
                                    make test-km3modules
                                """
                            } catch (e) { 
                                sendChatMessage("KM3Modules Test Suite Failed")
                                sendMail("KM3Modules Test Suite Failed")
                                throw e
                            }
                            try { 
                                sh """
                                    make clean
                                    make test-cov
                                """
                                if(DOCKER_NAME == MAIN_DOCKER) {
                                    step([$class: 'CoberturaPublisher',
                                            autoUpdateHealth: false,
                                            autoUpdateStability: false,
                                            coberturaReportFile: "reports/coverage.xml",
                                            failNoReports: false,
                                            failUnhealthy: false,
                                            failUnstable: false,
                                            maxNumberOfBuilds: 0,
                                            onlyStable: false,
                                            sourceEncoding: 'ASCII',
                                            zoomCoverageChart: false])
                                }
                            } catch (e) { 
                                sendChatMessage("Coverage Failed")
                                sendMail("Coverage Failed")
                                throw e
                            }
                        }

                        gitlabCommitStatus("Docs") {
                            try { 
                                sh """
                                    cd doc
                                    make html
                                """
                            } catch (e) { 
                                sendChatMessage("Building Docs Failed")
                                sendMail("Building Docs Failed")
                                throw e
                            }
                            publishHTML target: [
                                allowMissing: false,
                                alwaysLinkToLastBuild: false,
                                keepAll: true,
                                reportDir: "doc/_build/html",
                                reportFiles: 'index.html',
                                reportName: "Documentation"
                            ]
                        }
                    }
                }
    }


    stage("Reports") {
        step([$class: 'XUnitBuilder',
            thresholds: [
                [$class: 'SkippedThreshold', failureThreshold: '15'],
                [$class: 'FailedThreshold', failureThreshold: '0']],
            // thresholds: [[$class: 'FailedThreshold', unstableThreshold: '1']],
            tools: [[$class: 'JUnitType', pattern: 'reports/*.xml']]])
    }
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

