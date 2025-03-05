import aws_cdk as core
import aws_cdk.assertions as assertions

from src_aws_genai_con_carloscontreras.src_aws_genai_con_carloscontreras_stack import SrcAwsGenaiConCarloscontrerasStack

# example tests. To run these tests, uncomment this file along with the example
# resource in src_aws_genai_con_carloscontreras/src_aws_genai_con_carloscontreras_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = SrcAwsGenaiConCarloscontrerasStack(app, "src-aws-genai-con-carloscontreras")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
