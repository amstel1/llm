import re
from typing import List
from langchain_text_splitters import TextSplitter


class MarkdownTextSplitter(TextSplitter):
    def __init__(self, patterns = [r"\*\*(.*?)\*\*"]):
        self.patterns = patterns

    def split_text(self, text: str) -> List[str]:
        chunks = []
        matches = []
        for pattern in self.patterns:
            matches.extend(list(re.finditer(pattern, text)))
        n_matches = len(matches)
        for i, element in enumerate(matches):
            if i < n_matches-1:
                start_pos = element.span()[0]
                end_pos = matches[i+1].span()[0]
                chunks.append(text[start_pos:end_pos].strip())
            else:
                start_pos = element.span()[0]
                end_pos = -1
                chunks.append(text[start_pos:end_pos].strip())
        return chunks