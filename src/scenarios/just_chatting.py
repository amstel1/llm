import sys
sys.path.append('/home/amstel/llm/src')
from scenarios.scenario_router import ScenarioRouter
from scenarios.base import BaseScenario
from loguru import logger
from general_llm.llm_endpoint import call_generate_from_history_api
from typing import Iterable, Dict, List, Union, Optional, Any


class JustChattingScenario(BaseScenario):
    # we need to route at every step
    scenario_router = ScenarioRouter()

    def handle(self, user_query: Any, chat_history: Any, context: Any):
        assert 'previous_steps' in context
        assert isinstance(context.get('previous_steps'), list)
        response = call_generate_from_history_api(
            system_prompt='Отвечай точно.',
            chat_history=chat_history,
        )
        # context['current_step'] = 'chatting'
        context['previous_steps'].append('chatting')
        context['scenario_name'] = "just_chatting"
        return response, context
