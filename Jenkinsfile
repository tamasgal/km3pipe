#!groovy
import groovy.io.FileType
import static groovy.io.FileType.FILES

DOCKER_FILES_DIR = './dockerfiles'
CHAT_CHANNEL = '#km3pipe'
DEVELOPERS = ['tgal@km3net.de', 'mlotze@km3net.de']

properties([gitLabConnection('KM3NeT GitLab')])


def get_stages(dockerfile) {
    stages = {

        // Bug in Jenkins prevents using custom folder in docker.build
        def customImage = ''
        dir("${DOCKER_FILES_DIR}"){
            customImage = docker.build("km3pipe:${env.BUILD_ID}",
                                       "-f ${dockerfile} .")
        }

        customImage.inside("-u root:root") {

            // The following line causes a weird issue, where pip tries to 
            // install into /usr/local/... instead of the virtual env.
            // Any help figuring out what's happening is appreciated.
            //
            // def PYTHON_VENV = docker_image.replaceAll('[:.]', '') + 'venv'
            //
            // So we set it to 'venv' for all parallel builds now
            def DOCKER_NAME = dockerfile
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
                                        [$class: 'SkippedThreshold', failureThreshold: '5'],
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

    // def dockerfiles = ['py365']
    // TODO: figure out how to get the files dynamically
    // def dockerfiles = []
    // def dir = new File(DOCKER_FILES_DIR)
    // dir.eachFileRecurse (FileType.FILES) { file ->
    //     dockerfiles << file
    // }
    // dockerfiles = findFiles(glob: "${DOCKER_FILES_DIR}#<{(|")

    def dir = new File("${env.WORKSPACE}/${DOCKER_FILES_DIR});
    def dockerfiles = [];
    dir.traverse(type: FILES, maxDepth: 0) {
        dockerfiles.add(it)
    }

    def stages = [:]
    for (int i = 0; i < dockerfiles.size(); i++) {
        def dockerfile = dockerfiles[i]
        stages[dockerfile] = get_stages(dockerfile)
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
