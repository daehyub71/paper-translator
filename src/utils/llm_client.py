"""
LLM 클라이언트 모듈
OpenAI GPT-4o-mini를 사용한 번역 및 분석 기능 제공
"""
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()


class LLMClient:
    """OpenAI LLM 클라이언트"""

    _instance: Optional["LLMClient"] = None
    _client: Optional[OpenAI] = None
    _encoding: Optional[tiktoken.Encoding] = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """클라이언트 초기화"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
            )

        self._client = OpenAI(api_key=api_key)
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # tiktoken 인코딩 초기화 (gpt-4o-mini는 cl100k_base 사용)
        try:
            self._encoding = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def client(self) -> OpenAI:
        """OpenAI 클라이언트 반환"""
        if self._client is None:
            self._initialize()
        return self._client

    @property
    def model(self) -> str:
        """현재 모델명 반환"""
        return self._model

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return len(self._encoding.encode(text))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """메시지 리스트의 총 토큰 수 계산"""
        total = 0
        for message in messages:
            # 메시지 오버헤드 (~4 토큰)
            total += 4
            for key, value in message.items():
                total += self.count_tokens(str(value))
        # 답변 프라이밍 토큰
        total += 2
        return total

    def translate(
        self,
        text: str,
        terminology_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> dict:
        """
        텍스트 번역

        Args:
            text: 번역할 영어 텍스트
            terminology_prompt: 용어 매핑 프롬프트 (선택)
            temperature: 생성 온도 (낮을수록 일관성 높음)
            max_tokens: 최대 출력 토큰

        Returns:
            dict: {
                "translated_text": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
        """
        system_prompt = """당신은 AI/ML 분야 전문 번역가입니다.
주어진 영어 논문 텍스트를 한국어로 번역하세요.

번역 규칙:
1. 학술적이고 정확한 한국어를 사용하세요.
2. 전문 용어는 아래 용어집을 따르세요. 용어집에 없는 용어는 일반적인 번역을 사용하세요.
3. 수식, 코드, 참조(예: [1], Figure 1)는 원본 그대로 유지하세요.
4. 문장 구조를 자연스러운 한국어로 재구성하세요.
5. 원문의 의미를 정확히 전달하면서도 읽기 쉽게 번역하세요."""

        if terminology_prompt:
            system_prompt += f"\n\n## 용어집\n{terminology_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 텍스트를 한국어로 번역하세요:\n\n{text}"}
        ]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "translated_text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

    def analyze_term_changes(
        self,
        original_text: str,
        modified_text: str,
        known_terms: list[dict]
    ) -> dict:
        """
        번역문 변경에서 용어 변경 분석

        Args:
            original_text: 원본 번역문
            modified_text: 수정된 번역문
            known_terms: 알려진 용어 목록 [{"source": "...", "target": "..."}]

        Returns:
            dict: {
                "changes": [
                    {
                        "source_text": str,
                        "old_target": str,
                        "new_target": str,
                        "change_type": "update" | "add"
                    }
                ],
                "analysis": str
            }
        """
        terms_str = "\n".join([f"- {t['source']}: {t['target']}" for t in known_terms[:30]])

        system_prompt = """당신은 번역 품질 분석가입니다.
원본 번역문과 수정된 번역문을 비교하여 용어 변경을 분석하세요.

분석 방법:
1. 두 텍스트를 비교하여 변경된 부분을 찾습니다.
2. 변경이 전문 용어의 번역 변경인지 확인합니다.
3. 새로운 용어 번역이 추가되었는지 확인합니다.

출력 형식 (JSON):
{
    "changes": [
        {
            "source_text": "영어 원문 용어",
            "old_target": "기존 번역",
            "new_target": "새 번역",
            "change_type": "update"
        }
    ],
    "analysis": "변경 요약 설명"
}"""

        user_prompt = f"""## 알려진 용어
{terms_str}

## 원본 번역문
{original_text}

## 수정된 번역문
{modified_text}

위 두 번역문을 비교하여 용어 변경을 분석하고 JSON 형식으로 응답하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        import json
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {"changes": [], "analysis": "분석 실패"}

        return result

    def extract_paper_metadata(self, text: str) -> dict:
        """
        논문 텍스트에서 메타데이터 추출

        Args:
            text: 논문 텍스트 (주로 첫 부분)

        Returns:
            dict: {
                "title": str,
                "title_ko": str,
                "authors": list[str],
                "abstract": str,
                "domain": str
            }
        """
        system_prompt = """주어진 논문 텍스트에서 메타데이터를 추출하세요.

출력 형식 (JSON):
{
    "title": "영어 제목",
    "title_ko": "한국어 제목 번역",
    "authors": ["저자1", "저자2"],
    "abstract": "초록 전문 (영어)",
    "domain": "NLP | CV | RL | General 중 하나"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 논문 텍스트에서 메타데이터를 추출하세요:\n\n{text[:3000]}"}
        ]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        import json
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "title": "Unknown",
                "title_ko": "알 수 없음",
                "authors": [],
                "abstract": "",
                "domain": "General"
            }

        return result

    def completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False
    ) -> dict:
        """
        범용 completion 함수

        Args:
            messages: 메시지 리스트
            temperature: 생성 온도
            max_tokens: 최대 출력 토큰
            json_mode: JSON 응답 강제 여부

        Returns:
            dict: {
                "content": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
        """
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }


# 편의를 위한 전역 인스턴스
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """LLM 클라이언트 인스턴스 반환"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# 유틸리티 함수들
def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 계산 (단축 함수)"""
    return get_llm_client().count_tokens(text)


def translate_text(
    text: str,
    terminology_prompt: Optional[str] = None,
    temperature: float = 0.1
) -> dict:
    """텍스트 번역 (단축 함수)"""
    return get_llm_client().translate(text, terminology_prompt, temperature)
