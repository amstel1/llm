class InterestCalculator:
    annual_interest_rate = 0.1722
    def gpt4o(self, loan_sum, month_duration):
        payments = {}
        principal_payment = round((loan_sum / month_duration), 2)

        for month in range(1, month_duration + 1):
            remaining_balance = loan_sum - round((principal_payment * (month - 1)),2)
            interest_payment = round(remaining_balance * (self.annual_interest_rate / 12),2)
            total_payment = principal_payment + interest_payment
            payments[month] = total_payment

        min_monthly_payment = round(min(payments.values()),2)
        max_monthly_payment = round(max(payments.values()),2)
        total_payment_sum = round(sum(payments.values()),2)

        # {
        #     'interest_rate': self.annual_interest_rate,
        #     'month_duration': month_duration,
        #     'min_monthly_payment': min_monthly_payment,
        #     'max_monthly_payment': max_monthly_payment,
        #     'total_payment_sum': total_payment_sum
        # }
        loan_terms = f"""Этот товар выгодно взять в кредит!\n
Ежемесячный платеж составит от **{min_monthly_payment}** до **{max_monthly_payment}** руб.
(Общая сумма за весь срок: {total_payment_sum} руб.)"""
        return loan_terms

if __name__ == '__main__':
    # Example usage:
    loan_sum = 1250 #float(input("Enter the loan sum: "))
    month_duration = 24 # int(input("Enter the duration of the loan in months: "))
    calculator = InterestCalculator()
    loan_terms = calculator.gpt4o(loan_sum, month_duration)
    print(loan_terms)