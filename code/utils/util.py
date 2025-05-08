from typing import Tuple, List, Optional, Union, Any, Dict
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.api_keys import *
from functools import partial

# API clients
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Download NLTK data
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

# Configuration
m1_temperature = 0.7  # Temperature for first model
m2_temperature = 0.7  # Temperature for second model

# Default temperature for models
MODEL_TEMPERATURES = {
    "default": 0.7
}

# Global variable for sleep buffer
time_sleep_buffer = 0.1

# Model-specific settings for concurrency
MODEL_SETTINGS = {
    "ChatGPT": {"concurrent_limit": 10, "sleep_buffer": 1.0},
    "Claude": {"concurrent_limit": 5, "sleep_buffer": 1.0},
    "Llama": {"concurrent_limit": 1, "sleep_buffer": 1.5},  # Process sequentially
    "Grok": {"concurrent_limit": 15, "sleep_buffer": 0.8},
    "Gemini": {"concurrent_limit": 15, "sleep_buffer": 0.8},
    "Deepseek": {"concurrent_limit": 15, "sleep_buffer": 0.8},
    # Default values for any new model
    "default": {"concurrent_limit": 5, "sleep_buffer": 1.0}
}

# Retry configuration
DEFAULT_MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # Start with 2 seconds
BACKOFF_FACTOR = 2  # Exponential backoff
MAX_RETRY_DELAY = 60  # Cap at 60 seconds

# OpenAI API
def ChatGPT(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000) -> Tuple[str, Any]:
    """
    Get a response from gpt-4-turbo for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values like 0.8 will make the output more random
        max_tokens (int): Maximum number of tokens in the response
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    message = client.chat.completions.create(
      model="gpt-4-turbo",
      messages=[
        {"role": "user", "content": input_text}
      ],
      temperature=temperature,
      max_tokens=max_tokens,
      store=True
    )
    text_block = message.choices[0].message.content
    return text_block, message

# Claude API
def Claude(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000) -> Tuple[str, Any]:
    """
    Get a response from Claude for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values make the output more random
        max_tokens (int): Maximum number of tokens in the response
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    
    text_block = message.content[0].text
    return text_block, message

# Gemini API
def Gemini(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000, retries: int = 3, backoff_factor: int = 2) -> Tuple[str, Any]:
    """
    Get a response from Gemini for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values make the output more random
        max_tokens (int): Maximum number of tokens in the response
        retries (int): Number of retry attempts
        backoff_factor (int): Exponential backoff factor between retries
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    # Configure the model
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    
    # Retry logic
    for attempt in range(retries):
        try:
            response = model.generate_content(
                [input_text],
                generation_config=generation_config,
                safety_settings=[],
                stream=False
            )
            return response.text, response
        except Exception as e:
            print(f"Error generating for prompt: {e} (Attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                error_message = f"Error: {e}"
                return error_message, None
    
    return "Failed after multiple retries.", None

# Llama API
def Llama(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000) -> Tuple[str, Any]:
    """
    Get a response from Llama for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values make the output more random
        max_tokens (int): Maximum number of tokens in the response
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    client = OpenAI(
        api_key=LLAMA_API_KEY,
        base_url="https://api.llama-api.com"
    )
    
    message = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ],
        model="llama3.3-70b",
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    
    text_block = message.choices[0].message.content
    return text_block, message

# Grok API
def Grok(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000) -> Tuple[str, Any]:
    """
    Get a response from Grok for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values make the output more random
        max_tokens (int): Maximum number of tokens in the response
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    client = OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1"
    )
    
    message = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ],
        model="grok-2-latest",
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    
    text_block = message.choices[0].message.content
    return text_block, message

# Deepseek API
def Deepseek(input_text: str, temperature: float = MODEL_TEMPERATURES["default"], max_tokens: int = 3000) -> Tuple[str, Any]:
    """
    Get a response from Deepseek for the given input text.

    Args:
        input_text (str): The question or prompt
        temperature (float): Higher values make the output more random
        max_tokens (int): Maximum number of tokens in the response
    
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    message = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text
            }
        ],
        model="deepseek-chat",
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    
    text_block = message.choices[0].message.content
    return text_block, message

class TextProcessor:
    """
    A utility class for processing and analyzing text, particularly for sentence-level operations.
    """
    
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """
        Count the number of sentences in a text string.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            int: The number of sentences detected
        """
        if not isinstance(text, str):
            text = str(text)
        try:
            return len(sent_tokenize(text))
        except Exception as e:
            print(f"Error tokenizing: {text[:50]}... (Error: {e})")
            return 0

    @staticmethod
    def replace_nth_sentence(original_text: str, n: int, replacement_sentence: str) -> str:
        """
        Replace the nth sentence in the original text with the replacement sentence.
        
        Args:
            original_text (str): The original text with multiple sentences.
            n (int): The index of the sentence to replace (0-based).
            replacement_sentence (str): The new sentence to insert.
            
        Returns:
            str: The text with the nth sentence replaced.
        """
        # Use NLTK's sent_tokenize to properly split text into sentences
        sentences = sent_tokenize(original_text)
        
        # Only process if there are enough sentences
        if n <= len(sentences):
            sentences[n-1] = replacement_sentence.strip()
        
        # Join sentences back together with appropriate spacing
        return ' '.join(sentences)
    
    @staticmethod
    def get_nth_sentence(text: str, n: int) -> str:
        """
        Extract the nth sentence from a text string.
        
        Args:
            text (str): The input text
            n (int): The sentence number to extract (1-based index)
            
        Returns:
            str: The nth sentence if it exists, empty string otherwise
        """
        if not isinstance(text, str) or not isinstance(n, int):
            return ""
            
        try:
            sentences = sent_tokenize(text)
            if n <= len(sentences):
                return sentences[n - 1]
            return ""
        except Exception as e:
            print(f"Error extracting sentence: {e}")
            return ""
    
    @staticmethod
    def find_different_sentence(text_a: str, text_b: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the first sentence that differs between two texts.
        
        Args:
            text_a (str): First text for comparison
            text_b (str): Second text for comparison
            
        Returns:
            Tuple[Optional[int], Optional[str]]: The index of the differing sentence (1-based) 
                                               and the differing sentence from text_a
        """
        if not all(isinstance(x, str) for x in [text_a, text_b]):
            return None, None

        try:
            sentences_a = sent_tokenize(text_a)
            sentences_b = sent_tokenize(text_b)

            min_len = min(len(sentences_a), len(sentences_b))

            for i in range(min_len):
                if sentences_a[i].strip() != sentences_b[i].strip():
                    return i + 1, sentences_a[i]

            if len(sentences_a) != len(sentences_b):
                return min_len + 1, None
            return -1, None

        except Exception as e:
            print(f"Error comparing texts: {e}")
            return None, None

    @staticmethod
    def clean_text(text):
        """
        Clean text by fixing common formatting issues like double periods
        and improper punctuation around periods.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            return text
            
        # Fix any double periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Fix missing spaces after periods
        text = re.sub(r'\.([A-Z])', '. \\1', text)
        
        # Fix punctuation immediately before periods (like ?.)
        # Keep only the strongest punctuation (., !, ?)
        text = re.sub(r'([.!?])\.', r'\1', text)
        
        # Fix punctuation immediately after periods (like .?)
        # For cases where a period is followed by another punctuation mark,
        # ensure there's a space in between
        text = re.sub(r'\.([.!?,;:])', r'. \1', text)
        
        # Final cleanup to ensure no duplicate spaces
        text = re.sub(r' +', ' ', text)
        
        return text
    
def convert_to_ordinal_text(df, column_name):
    """
    Convert numbers in a pandas DataFrame column to their ordinal text equivalents.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the column to convert
        column_name (str): The name of the column to convert
        
    Returns:
        pandas.Series: A series with the converted ordinal text values
    """
    # Create a mapping dictionary of numbers to ordinal text
    ordinal_mapping = {
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth',
        6: 'sixth',
        7: 'seventh',
        8: 'eighth',
        9: 'ninth',
        10: 'tenth',
        11: 'eleventh',
        12: 'twelfth',
        13: 'thirteenth',
        14: 'fourteenth',
        15: 'fifteenth',
        16: 'sixteenth',
        17: 'seventeenth',
        18: 'eighteenth',
        19: 'nineteenth',
        20: 'twentieth'
    }
    return df[column_name].map(ordinal_mapping)

def select_random_integers(total_count=50, selection_count=20):
    """
    Randomly select non-overlapping integers from a range.
    
    Args:
        total_count (int): The upper bound of the range (starting from 1)
        selection_count (int): Number of integers to select
    
    Returns:
        list: A list of randomly selected integers
    """
    if selection_count > total_count:
        selection_count = total_count
        
    # Create a list of all possible integers
    all_integers = list(range(1, total_count + 1))
    
    # Randomly shuffle the list
    random.shuffle(all_integers)
    
    # Return the first selection_count elements
    return all_integers[:selection_count]

def add_confidence_interval(df, n_samples=100, confidence_level=0.95):
    """
    Add confidence interval for Bernoulli-distributed recognition accuracy
    Using the normal approximation to binomial (valid for large n)

    Args:
        df (pandas.DataFrame): DataFrame with Recognition_accuracy column
        n_samples (int): Number of observations used to compute the accuracy
        confidence_level (float): Desired confidence level (default 0.95 for 95% CI)

    Returns:
        pandas.DataFrame: DataFrame with added columns: ci_lower_bound, ci_upper_bound, confidence_interval
    """
    if 'Recognition_accuracy' not in df.columns:
        raise ValueError("DataFrame must contain 'Recognition_accuracy' column")

    p = df['Recognition_accuracy']  # Success probability (accuracy)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Standard error for Bernoulli distribution
    std_error = np.sqrt((p * (1 - p)) / n_samples)

    # Calculate confidence interval
    margin_of_error = z_score * std_error

    # Add separate columns for lower and upper bounds
    df['ci_lower_bound'] = (p - margin_of_error).round(4)
    df['ci_upper_bound'] = (p + margin_of_error).round(4)

    # Keep the string representation for backward compatibility
    df['confidence_interval'] = df.apply(
        lambda row: f"({row['ci_lower_bound']:.2f}, {row['ci_upper_bound']:.2f})",
        axis=1
    )

    return df

def call_model_safely(model_func, prompt, temperature=MODEL_TEMPERATURES["default"], max_tokens=3000):
    """
    Call the model with appropriate rate limiting and retries.
    
    Args:
        model_func (callable): The model function to call
        prompt (str): The input text/prompt
        temperature (float): Temperature setting for the model
        max_tokens (int): Maximum tokens for the response
        
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    max_retries = 3
    retry_count = 0
    model_name = model_func.__name__
    
    # Set wait times with increase for all models
    base_sleep = 0.3 if model_name in ["Llama", "Claude"] else 0.1
    
    while retry_count < max_retries:
        try:
            # Call the model
            result = model_func(prompt, temperature=temperature, max_tokens=max_tokens)
            
            # Sleep after successful call
            time.sleep(base_sleep + time_sleep_buffer)
            return result
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            # Handle 409 errors specially, with increased wait times
            if "409" in error_msg or "Concurrency conflict" in error_msg:
                wait_time = base_sleep * (2 ** retry_count) + time_sleep_buffer
            else:
                wait_time = base_sleep * (1.5 ** retry_count) + time_sleep_buffer
                
            time.sleep(wait_time)
            
            if retry_count == max_retries:
                return f"Failed after {max_retries} attempts: {e}", None
    
    return "Failed", None

async def async_model_call(model_func, prompt, temperature=MODEL_TEMPERATURES["default"], max_tokens=3000, is_llama=False, llama_semaphore=None):
    """
    Asynchronous wrapper for model call.
    
    Args:
        model_func (callable): The model function to call
        prompt (str): The input text/prompt
        temperature (float): Temperature setting for the model
        max_tokens (int): Maximum tokens for the response
        is_llama (bool): Whether this is a Llama model call (requires special handling)
        llama_semaphore (asyncio.Semaphore): Optional semaphore for Llama API calls
        
    Returns:
        Tuple[str, Any]: The response text and the full response object
    """
    loop = asyncio.get_event_loop()
    
    # If it's a Llama call, use the dedicated semaphore to control concurrency
    if is_llama and llama_semaphore:
        async with llama_semaphore:
            # Add retry logic specific for Llama
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second delay
            
            for attempt in range(max_retries):
                try:
                    with ThreadPoolExecutor() as pool:
                        result = await loop.run_in_executor(
                            pool, 
                            partial(model_func, prompt, temperature)
                        )
                    # Successfully got a result, return it
                    return result
                except Exception as e:
                    if "Concurrency conflict" in str(e) and attempt < max_retries - 1:
                        # If we got a concurrency error and have retries left
                        print(f"Llama API concurrency conflict, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        # Exponential backoff for retry delay
                        retry_delay *= 2
                    else:
                        # Reraise the exception if it's not a concurrency error or no retries left
                        raise
    else:
        # Use the standard model call for non-Llama models
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                lambda: call_model_safely(model_func, prompt, temperature, max_tokens)
            )

# Helper function to safely convert model response to storable string
def safe_serialize_result(result):
    """Convert model result to a safely serializable string"""
    try:
        # Try JSON serialization first (handles most complex structures)
        import json
        return json.dumps(result)
    except (TypeError, ValueError, OverflowError):
        # If that fails, use a simple string representation
        return str(result)

# Helper function to safely extract message text from result
def extract_message_from_result(result):
    """Safely extract the message content from various result formats"""
    try:
        if isinstance(result, tuple) and len(result) > 0:
            return str(result[0])
        elif isinstance(result, list) and len(result) > 0:
            return str(result[0])
        elif isinstance(result, dict):
            if 'message' in result:
                return str(result['message'])
            elif 'content' in result:
                return str(result['content'])
            elif 'text' in result:
                return str(result['text'])
            # Return the first value in the dict as a fallback
            elif len(result) > 0:
                return str(next(iter(result.values())))
        # For primitive types or as a fallback
        return str(result)
    except Exception as e:
        print(f"Error extracting message from result: {e}")
        return str(result)

# Helper function to check if response indicates a rate limit error
def is_rate_limit_error(response):
    """Check if the response indicates a rate limit error"""
    if isinstance(response, tuple) and len(response) > 0:
        response_text = response[0]
        return isinstance(response_text, str) and any(err in response_text.lower() for err in ["rate_limit", "429", "too many request"])
    return False

# Helper function to check for any API error
def is_api_error(response):
    """Check if the response indicates any API error"""
    if isinstance(response, tuple) and len(response) > 0:
        response_text = response[0]
        return isinstance(response_text, str) and "error" in response_text.lower()
    return False

# Helper function to extract error code from response
def extract_error_code(response):
    """Extract HTTP error code from error response if available"""
    if isinstance(response, tuple) and len(response) > 0:
        response_text = response[0]
        if isinstance(response_text, str):
            match = re.search(r'(\d{3})', response_text)
            if match:
                return match.group(1)
    return "unknown"

# Enhanced async model call with retry logic
async def async_model_call_with_retry(model_func, prompt, temperature=MODEL_TEMPERATURES["default"], 
                                     prompt_id=None, max_retries=DEFAULT_MAX_RETRIES, 
                                     initial_retry_delay=INITIAL_RETRY_DELAY,
                                     backoff_factor=BACKOFF_FACTOR,
                                     max_retry_delay=MAX_RETRY_DELAY,
                                     is_llama=False, llama_semaphore=None):
    """
    Call model with retry logic for rate limit errors.
    
    Args:
        model_func: The model function to call
        prompt: The prompt to process
        temperature: Temperature setting for the model
        prompt_id: Identifier for the prompt (for error logging)
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        max_retry_delay: Maximum delay in seconds
        is_llama: Whether this is a Llama model (requires special handling)
        llama_semaphore: Optional semaphore for Llama API calls
        
    Returns:
        Tuple containing the model response and additional metadata
    """
    model_name = model_func.__name__
    retries = 0
    retry_delay = initial_retry_delay
    
    while True:
        try:
            # Use the async_model_call with llama handling if specified
            result = await async_model_call(model_func, prompt, temperature, 
                                           is_llama=is_llama, 
                                           llama_semaphore=llama_semaphore)
            
            # Check if the result indicates an API error
            if is_api_error(result) and retries < max_retries:
                retries += 1
                error_type = "rate limit" if is_rate_limit_error(result) else "API"
                error_code = extract_error_code(result)
                
                print(f"{error_type} error (code {error_code}) for {model_name}. Retry {retries}/{max_retries} after {retry_delay}s")
                await asyncio.sleep(retry_delay)
                
                # Apply exponential backoff with jitter for next retry
                retry_delay = min(retry_delay * backoff_factor * (1 + random.random() * 0.2), max_retry_delay)
            else:
                # No error or max retries reached
                if retries >= max_retries and is_api_error(result):
                    print(f"Max retries ({max_retries}) reached for {model_name}. Giving up.")
                return result
                
        except Exception as e:
            error_str = str(e)
            if retries < max_retries:
                retries += 1
                print(f"Error calling {model_name}: {error_str}. Retry {retries}/{max_retries} after {retry_delay}s")
                await asyncio.sleep(retry_delay)
                # Apply exponential backoff with jitter for next retry
                retry_delay = min(retry_delay * backoff_factor * (1 + random.random() * 0.2), max_retry_delay)
            else:
                # Max retries reached, return error
                print(f"Max retries ({max_retries}) reached for {model_name}. Giving up.")
                return (f"Error: {error_str}", {"error": error_str})

# Function to process a batch of prompts with retries
async def process_batch(model_func, prompts, temperature=MODEL_TEMPERATURES["default"], 
                        prompt_ids=None, is_llama=False, llama_semaphore=None, 
                        time_sleep_buffer=0.1, max_retries=DEFAULT_MAX_RETRIES):
    """
    Process a batch of prompts asynchronously with the given model.
    
    Args:
        model_func: The model function to call
        prompts: List of prompts to process
        temperature: Temperature setting for the model
        prompt_ids: Optional list of identifiers for the prompts (for error logging)
        is_llama: Whether this is a Llama model (requires special handling)
        llama_semaphore: Optional semaphore for Llama API calls
        time_sleep_buffer: Buffer time between API calls
        max_retries: Maximum number of retry attempts for each prompt
        
    Returns:
        List of tuples containing model outputs
    """
    if prompt_ids is None:
        prompt_ids = [None] * len(prompts)
        
    # For Llama, process sequentially to avoid concurrency issues
    if is_llama:
        results = []
        for prompt, prompt_id in zip(prompts, prompt_ids):
            result = await async_model_call_with_retry(
                model_func, prompt, temperature, 
                prompt_id=prompt_id, max_retries=max_retries,
                is_llama=True, llama_semaphore=llama_semaphore
            )
            results.append(result)
            # Add a longer delay between Llama API calls
            await asyncio.sleep(0.5)
        return results
    else:
        # For other models, process in parallel using the retry function
        tasks = [
            async_model_call_with_retry(
                model_func, prompt, temperature, 
                prompt_id=prompt_id, max_retries=max_retries
            ) 
            for prompt, prompt_id in zip(prompts, prompt_ids)
        ]
        return await asyncio.gather(*tasks)

# Save checkpoint function for long-running processes
def save_checkpoint(df, checkpoint_file, completed_indices):
    """Save a checkpoint of the current progress"""
    try:
        # Create a copy with only the completed rows
        checkpoint_df = df.copy()
        
        # Mark which rows have been processed
        checkpoint_df['completed'] = False
        for idx in completed_indices:
            if idx < len(checkpoint_df):
                checkpoint_df.at[idx, 'completed'] = True
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        # Save checkpoint
        checkpoint_df.to_csv(checkpoint_file, index=False, sep='|')
        print(f"Checkpoint saved: {checkpoint_file} ({len(completed_indices)}/{len(df)} rows processed)")
    except Exception as e:
        print(f"Warning: Error saving checkpoint: {e}")

# Create a model registry for easier access to all models
MODEL_REGISTRY = {
    "claude": Claude,
    "chatgpt": ChatGPT,
    "gemini": Gemini,
    "llama": Llama,
    "grok": Grok,
    "deepseek": Deepseek
}

def get_model(model_name):
    """
    Get a model function by name.
    
    Args:
        model_name (str): The name of the model (case-insensitive)
        
    Returns:
        callable: The model function
    """
    return MODEL_REGISTRY.get(model_name.lower(), None)

# General function to process a model's prompts with optimized handling
async def process_model_prompts(model_func, prompts, temperature=None, 
                               prompt_ids=None, output_processor=None,
                               batch_size=5, progress_desc=None):
    """
    Process a list of prompts with a model, handling model-specific optimizations.
    
    Args:
        model_func: The model function to call
        prompts: List of prompts to process
        temperature: Temperature setting (if None, use model's default)
        prompt_ids: Optional list of identifiers for the prompts
        output_processor: Optional function to process each output
        batch_size: Size of batches to process at once
        progress_desc: Description for the progress bar
        
    Returns:
        List of processed results
    """
    model_name = model_func.__name__
    if temperature is None:
        temperature = MODEL_TEMPERATURES["default"]
        
    # Get model-specific settings
    settings = MODEL_SETTINGS.get(model_name, MODEL_SETTINGS["default"])
    concurrent_limit = settings["concurrent_limit"]
    sleep_buffer = settings["sleep_buffer"]
    
    # Create description for progress bar
    if progress_desc is None:
        progress_desc = f"Processing {model_name}"
    
    # Flag for special handling models
    is_llama = model_name == "Llama"
    
    # Create semaphore for concurrency control
    if concurrent_limit > 1:
        semaphore = asyncio.Semaphore(concurrent_limit)
    else:
        # For sequential processing, use a semaphore with limit 1
        semaphore = asyncio.Semaphore(1)
    
    # Create a list to store results
    results = []
    
    # For models that need sequential processing or have very low concurrency limits
    if concurrent_limit <= 2 or model_name in ["Llama", "Claude"]:
        from tqdm import tqdm
        for i, prompt in enumerate(tqdm(prompts, desc=progress_desc)):
            prompt_id = prompt_ids[i] if prompt_ids else None
            
            # Process with retry logic
            result = await async_model_call_with_retry(
                model_func, prompt, temperature,
                prompt_id=prompt_id, is_llama=is_llama,
                llama_semaphore=semaphore if is_llama else None
            )
            
            # Process the result if a processor is provided
            if output_processor:
                result = output_processor(result)
                
            results.append(result)
            
            # Add variable sleep with randomization to avoid synchronized requests
            await asyncio.sleep(sleep_buffer + random.random() * 0.5)
    else:
        # For models that can handle higher concurrency, process in batches
        from tqdm import tqdm
        
        # Process in batches for better throughput and progress tracking
        for i in tqdm(range(0, len(prompts), batch_size), desc=progress_desc):
            batch_prompts = prompts[i:i+batch_size]
            batch_ids = prompt_ids[i:i+batch_size] if prompt_ids else None
            
            # Process the batch
            batch_results = await process_batch(
                model_func, batch_prompts, temperature,
                prompt_ids=batch_ids, is_llama=is_llama,
                llama_semaphore=semaphore if is_llama else None,
                time_sleep_buffer=sleep_buffer
            )
            
            # Process the results if a processor is provided
            if output_processor:
                batch_results = [output_processor(result) for result in batch_results]
                
            results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limiting
            await asyncio.sleep(sleep_buffer * 0.5 + random.random() * 0.5)
    
    return results


def find_closest_sentence(text: str, target: str, threshold: float = 0.3) -> Tuple[int, float]:
    """
    Finds the sentence in text most similar to target using string similarity.

    Args:
        text: Source text containing multiple sentences
        target: Single sentence to match
        threshold: Minimum similarity threshold

    Returns:
        Tuple of (sentence position (1-based), similarity score)
    """
    try:
        sentences = sent_tokenize(text)
        scores = [(i+1, SequenceMatcher(None, s.lower(), target.lower()).ratio())
                 for i, s in enumerate(sentences)]
        best_match = max(scores, key=lambda x: x[1])
        return best_match if best_match[1] >= threshold else (0, 0.0)
    except Exception as e:
        print(f"Error matching sentences: {e}")
        return (0, 0.0)
    

def validate_sentence_matches(
    df: pd.DataFrame,
    sentences_col: str = 'step2_output',
    one_sentence_col: str = 'step3_output_message_only'
) -> Tuple[float, pd.DataFrame]:
    """
    Identifies matched sentences and validates against expected positions.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        sentences_col (str): Column name for step 2 sentences output
        one_sentence_col (str): Column name for step 3 sentence output
    
    Returns:
        tuple: (accuracy, processed_dataframe)
    """
    df = df.copy()
    
    # Find best matching sentences
    matches = df.apply(lambda row: find_closest_sentence(
        row[sentences_col], row[one_sentence_col]), axis=1)
    
    df['step3_output_nth_sentence'] = [m[0] for m in matches]
    df['match_confidence'] = [m[1] for m in matches]
    
    # Add the new column to show the closest sentence position
    df['step3_output_nth_sentence_result'] = df['step3_output_nth_sentence'].astype(int)
    
    # Calculate accuracy
    matches_expected = sum(df['step2_random_sent_num'] == df['step3_output_nth_sentence'])
    accuracy = matches_expected / len(df)
    print(f"Match accuracy: {accuracy:.2%}")
    
    return accuracy, df

def get_balanced_random_assignments(df, count_column, random_state=None):
    """
    Creates balanced random assignments for a dataframe column containing count values.
    
    This function ensures each possible integer value (from 1 to the count value) appears 
    with equal frequency in the random assignments, unlike random.randint which may not 
    distribute outcomes evenly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the count column
    count_column : str
        Name of the column containing positive integer counts
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        A dictionary mapping each row index to its balanced random assignment
    """
    import random
    import pandas as pd
    
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    
    # Get valid rows and their counts
    valid_rows = df[pd.notna(df[count_column]) & (df[count_column] > 0)]
    
    # Dictionary to store the balanced random assignment for each row index
    assignments = {}
    
    # Group by count value for efficiency
    for count_value, group in valid_rows.groupby(count_column):
        count = int(count_value)  # Convert to int to handle potential float values
        if count <= 0:
            continue
            
        # All possible values from 1 to count
        possible_values = list(range(1, count + 1))
        
        # Number of rows with this count value
        num_rows = len(group)
        
        # Create a balanced list by repeating and shuffling
        # Calculate how many full cycles we need
        full_cycles = num_rows // count
        remainder = num_rows % count
        
        # Create the balanced list
        balanced_list = possible_values * full_cycles
        
        # Add remainder values if needed (this maintains balance as much as possible)
        if remainder > 0:
            # Take a random sample without replacement for the remainder
            balanced_list.extend(random.sample(possible_values, remainder))
        
        # Shuffle the list to randomize the order while maintaining balance
        random.shuffle(balanced_list)
        
        # Assign values to each row
        for i, (idx, _) in enumerate(group.iterrows()):
            assignments[idx] = balanced_list[i]
    
    return assignments

def apply_balanced_random(df, count_column, output_column, random_state=None):
    """
    Applies balanced random values to a dataframe based on a count column.
    
    For each positive integer value in the count column, this function assigns
    a random integer between 1 and that value, ensuring equal distribution of
    all possible outcomes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to modify
    count_column : str
        Name of the column containing positive integer counts
    output_column : str
        Name of the column where balanced random values will be stored
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        The modified dataframe with the new balanced random column
    """
    # Get balanced random assignments
    assignments = get_balanced_random_assignments(df, count_column, random_state)
    
    # Apply assignments to the dataframe
    df = df.copy()  # Avoid modifying the original dataframe
    
    # Initialize output column with None
    df[output_column] = None
    
    # Fill in the assignments
    for idx, value in assignments.items():
        df.at[idx, output_column] = value
    
    return df