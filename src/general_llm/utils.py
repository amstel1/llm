
class ChatHistory:
    @staticmethod
    def truncate_exclude_last(chat_history: list[dict], n: int=4):
        '''
        Truncates the chat history to last n pairs.
        Only roles in (user, assisstant) are taken in consideration.
        Note that number can be not odd
        :param n:
        :return:
        '''
        return [x for x in chat_history if x.get('role') != 'html'][-n-1:-1]  # убираем из истории чата текущую (последнюю) реплику

    def truncate_include_last(chat_history: list[dict], n: int=4):
        '''
        Truncates the chat history to last n pairs.
        Only roles in (user, assisstant) are taken in consideration.
        Note that number can be not odd
        :param n:
        :return:
        '''
        return [x for x in chat_history if x.get('role') != 'html'][-n:]  # убираем из истории чата текущую (последнюю) реплику