"""Handlers for all OLS-related REST API endpoints."""

import logging

from fastapi import APIRouter, HTTPException, status
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ols import constants
from ols.app.models.models import LLMRequest
from ols.app.utils import Utils
from ols.src.llms.llm_loader import LLMLoader
from ols.src.query_helpers.docs_summarizer import DocsSummarizer
from ols.src.query_helpers.question_validator import QuestionValidator
from ols.src.query_helpers.yaml_generator import YamlGenerator
from ols.utils import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query")
def conversation_request(llm_request: LLMRequest) -> LLMRequest:
    """Handle conversation requests for the OLS endpoint.

    Args:
        llm_request: The request containing a query and conversation ID.

    Returns:
        Response containing the processed information.
    """
    # Initialize variables
    previous_input = None
    conversation = llm_request.conversation_id

    # Generate a new conversation ID if not provided
    if conversation is None:
        conversation = Utils.get_suid()
        logger.info(f"{conversation} New conversation")
    else:
        previous_input = config.conversation_cache.get(conversation)
        logger.info(f"{conversation} Previous conversation input: {previous_input}")

    llm_response = LLMRequest(query=llm_request.query, conversation_id=conversation)

    # Log incoming request
    logger.info(f"{conversation} Incoming request: {llm_request.query}")

    # Validate the query
    question_validator = QuestionValidator()
    validation_result = question_validator.validate_question(
        conversation, llm_request.query
    )

    if validation_result[0] == constants.INVALID:
        logger.info(f"{conversation} Question is not about k8s/ocp, rejecting")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "response": "Sorry, I can only answer questions about "
                "OpenShift and Kubernetes. This does not look "
                "like something I know how to handle."
            },
        )

    if validation_result[0] == constants.VALID:
        logger.info(f"{conversation} Question is about k8s/ocp")
        question_type = validation_result[1]

        # check if question type is from known categories
        if question_type not in {constants.NOYAML, constants.YAML}:
            # not known question type has been detected
            logger.error(f"Unknown question type {question_type}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"response": "Internal server error. Please try again."},
            )

        if question_type == constants.NOYAML:
            logger.info(
                f"{conversation} Question is not about yaml, sending for generic info"
            )

            # Summarize documentation
            docs_summarizer = DocsSummarizer()
            llm_response.response, _ = docs_summarizer.summarize(
                conversation, llm_request.query
            )

            return llm_response

        elif question_type == constants.YAML:
            logger.info(
                f"{conversation} Question is about yaml, sending to the YAML generator"
            )
            yaml_generator = YamlGenerator()
            generated_yaml = yaml_generator.generate_yaml(
                conversation, llm_request.query, previous_input
            )

            if generated_yaml == constants.SOME_FAILURE:
                raise HTTPException(
                    status_code=500,
                    detail={"response": "Internal server error. Please try again."},
                )

            # Further processing of YAML response (filtering, cleaning, linting, RAG, etc.)

            llm_response.response = generated_yaml

            if config.conversation_cache is not None:
                config.conversation_cache.insert_or_append(
                    conversation,
                    llm_request.query + "\n\n" + str(llm_response.response or ""),
                )
            return llm_response
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"response": "Internal server error. Please try again."},
    )


@router.post("/debug/query")
def conversation_request_debug_api(llm_request: LLMRequest) -> LLMRequest:
    """Handle requests for the base LLM completion endpoint.

    Args:
        llm_request: The request containing a query.

    Returns:
        Response containing the processed information.
    """
    if llm_request.conversation_id is None:
        conversation = Utils.get_suid()
    else:
        conversation = llm_request.conversation_id

    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logger.info(f"{conversation} New conversation")
    logger.info(f"{conversation} Incoming request: {llm_request.query}")

    bare_llm = LLMLoader(
        config.ols_config.default_provider,
        config.ols_config.default_model,
    ).llm

    prompt = PromptTemplate.from_template("{query}")
    llm_chain = LLMChain(llm=bare_llm, prompt=prompt, verbose=True)
    response = llm_chain(inputs={"query": llm_request.query})

    logger.info(f"{conversation} Model returned: {response}")

    llm_response.response = response["text"]

    return llm_response
